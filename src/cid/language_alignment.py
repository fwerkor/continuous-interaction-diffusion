from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cid.model.ar import (
    AR_CID_MODEL_TYPES,
    bidirectional_ar_hidden_states,
    prepare_ar_backbone_for_cid,
)
from cid.stage_c import StageCExample, load_stage_c_examples


@dataclass(frozen=True)
class LASample:
    input_ids: Tensor
    targets: Tensor
    content_mask: Tensor
    masked_fraction: float


def build_la_sample(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    mask_probability: float,
    generator: torch.Generator,
    canvas_tokens: int,
    max_answer_tokens: int,
    use_native_template: bool = False,
    supervise_eos_probability: float = 0.25,
) -> LASample:
    if not 0.0 < mask_probability <= 1.0:
        raise ValueError("mask_probability must lie in (0, 1]")
    if canvas_tokens < 2 or max_answer_tokens < 2:
        raise ValueError("Display canvas and answer budget must contain answer and EOS")
    if not 0.0 <= supervise_eos_probability <= 1.0:
        raise ValueError("supervise_eos_probability must lie in [0, 1]")
    if use_native_template:
        if tokenizer.chat_template is None:
            raise ValueError("teacher requires a native chat template")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    prefix = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids or len(answer_ids) + 1 > min(canvas_tokens, max_answer_tokens):
        raise ValueError("answer outside Display token budget")
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(tokenizer.eos_token_id)
    if mask_id == eos_id:
        raise ValueError("MASK and EOS must have different token IDs")

    # Match the CID preallocated canvas: all latent positions are MASK, last
    # position is a provisional EOS; the target EOS may be at an earlier index.
    canvas = [mask_id] * canvas_tokens
    canvas[-1] = eos_id
    targets = torch.full((canvas_tokens,), -100, dtype=torch.long)
    draws = torch.rand((len(answer_ids),), generator=generator)
    masked = draws < mask_probability
    if not bool(masked.any()):
        masked[int(draws.argmin())] = True
    for index, token in enumerate(answer_ids):
        if masked[index]:
            targets[index] = token
        else:
            canvas[index] = token
    eos_position = len(answer_ids)
    if (
        eos_position < canvas_tokens - 1
        and float(torch.rand((), generator=generator)) < supervise_eos_probability
    ):
        targets[eos_position] = eos_id
    elif eos_position < canvas_tokens - 1:
        canvas[eos_position] = eos_id
    # No loss on the fixed terminal sentinel (it already is EOS).
    content_mask = targets.ne(-100) & torch.arange(canvas_tokens).lt(len(answer_ids))
    return LASample(
        input_ids=torch.tensor([*prefix, *canvas], dtype=torch.long),
        targets=targets,
        content_mask=content_mask,
        masked_fraction=float(masked.float().mean()),
    )


def la_objective(
    student_logits: Tensor,
    targets: Tensor,
    *,
    teacher_logits: Tensor | None = None,
    content_mask: Tensor | None = None,
    distillation_weight: float = 0.0,
    temperature: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor]:
    if not distillation_weight >= 0.0 or temperature <= 0.0:
        raise ValueError("invalid distillation hyperparameters")
    ce = F.cross_entropy(student_logits.float(), targets, ignore_index=-100)
    kl = ce.new_zeros(())
    if teacher_logits is not None and distillation_weight:
        if teacher_logits.shape != student_logits.shape:
            raise ValueError("teacher and student answer logits must align")
        if content_mask is None or not bool(content_mask.any()):
            raise ValueError("distillation requires masked answer content")
        selected_student = student_logits[content_mask].float() / temperature
        selected_teacher = teacher_logits.detach()[content_mask].float() / temperature
        kl = F.kl_div(
            F.log_softmax(selected_student, dim=-1),
            F.softmax(selected_teacher, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
    return ce + distillation_weight * kl, ce.detach(), kl.detach()


class LABackbone(nn.Module):
    def __init__(self, backbone: nn.Module, *, autoregressive: bool):
        super().__init__()
        self.backbone = backbone
        self.autoregressive = autoregressive

    def forward(self, ids: Tensor) -> Tensor:
        if not self.autoregressive:
            return self.backbone(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        decoder = self.backbone.get_decoder()
        hidden = bidirectional_ar_hidden_states(
            decoder,
            inputs_embeds=self.backbone.get_input_embeddings()(ids),
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            position_ids=torch.arange(ids.shape[-1], device=ids.device)[None, :].expand_as(ids),
        )
        return self.backbone.get_output_embeddings()(hidden)


def load_la_backbone(path: str, *, dtype: torch.dtype) -> tuple[LABackbone, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    family = str(config.model_type)
    ar = family in AR_CID_MODEL_TYPES
    if ar:
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype, attn_implementation="sdpa"
        )
        prepare_ar_backbone_for_cid(model, tokenizer)
    elif family in {"illada", "llada"}:
        # Both are noncausal diffusion models exposed through
        # AutoModelForCausalLM; they do not use AR attention conversion.
        from cid.model.illada import ILLADA_MASK_TOKEN_ID, LLADA_MOE_MASK_TOKEN_ID

        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if family == "llada":
            kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        expected_mask = ILLADA_MASK_TOKEN_ID if family == "illada" else LLADA_MOE_MASK_TOKEN_ID
        if tokenizer.mask_token_id is None:
            native_token = tokenizer.convert_ids_to_tokens(expected_mask)
            if tokenizer.convert_tokens_to_ids(native_token) != expected_mask:
                raise ValueError("diffusion backbone MASK token cannot be recovered")
            tokenizer.mask_token = native_token
        if tokenizer.mask_token_id != expected_mask:
            raise ValueError("diffusion backbone MASK token differs from CID runtime")
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype
        )
        if tokenizer.mask_token_id is None:
            raise ValueError(f"non-AR backbone {family} must define a mask token")
    if tokenizer.eos_token_id is None:
        raise ValueError("LA requires an EOS token")
    return LABackbone(model, autoregressive=ar), tokenizer


def la_forward(model: LABackbone, sample: LASample, *, device: torch.device) -> Tensor:
    ids = sample.input_ids[None, :].to(device)
    return model(ids)[0, -sample.targets.numel() :, :]


def save_la_backbone(
    model: LABackbone, tokenizer: Any, directory: Path, metadata: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    (directory / "la_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def evaluate_la(
    model: LABackbone,
    tokenizer: Any,
    examples: list[StageCExample],
    *,
    device: torch.device,
    canvas_tokens: int,
    max_answer_tokens: int,
    seed: int,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            try:
                sample = build_la_sample(
                    tokenizer,
                    example.prompt,
                    example.answer,
                    mask_probability=0.75,
                    generator=generator,
                    canvas_tokens=canvas_tokens,
                    max_answer_tokens=max_answer_tokens,
                )
            except ValueError:
                continue
            if sample.input_ids.numel() > model.backbone.config.max_position_embeddings:
                continue
            logits = la_forward(model, sample, device=device)
            targets = sample.targets.to(device)
            loss = F.cross_entropy(logits.float(), targets, ignore_index=-100)
            losses.append(float(loss))
    model.train()
    if not losses:
        raise ValueError("no evaluation examples fit LA token budgets")
    return sum(losses) / len(losses)


def train_language_alignment(args: Any) -> None:
    if args.max_steps <= 0 or args.learning_rate <= 0 or args.grad_accum <= 0:
        raise ValueError(
            "optimizer steps, learning rate and gradient accumulation must be positive"
        )
    if args.distillation_weight < 0.0 or args.temperature <= 0.0:
        raise ValueError("invalid teacher parameters")
    if not 0.0 < args.min_mask_probability <= 1.0:
        raise ValueError("minimum LA mask probability must lie in (0, 1]")
    if args.log_every <= 0 or args.save_every < 0:
        raise ValueError("LA log interval must be positive and save interval nonnegative")
    if args.canvas_tokens < 2 or args.max_answer_tokens < 2:
        raise ValueError("invalid LA Display canvas or answer budget")
    if args.distillation_weight and not args.native_teacher:
        raise ValueError("native teacher model is required for distillation")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model, tokenizer = load_la_backbone(args.model, dtype=torch.float32)
    model.to(device).train()
    teacher = None
    teacher_tokenizer = None
    if args.native_teacher:
        teacher, teacher_tokenizer = load_la_backbone(args.native_teacher, dtype=torch.float32)
        if teacher.autoregressive != model.autoregressive:
            raise ValueError("teacher must have the same backbone architecture")
        if len(teacher_tokenizer) != len(tokenizer):
            raise ValueError("teacher and student tokenizers must have identical vocabulary sizes")
        if teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("teacher and student tokenizers must have identical token IDs")
        teacher.to(device).eval().requires_grad_(False)
    train = load_stage_c_examples(args.data, max_examples=args.max_examples)
    validation = (
        load_stage_c_examples(args.validation_data, max_examples=args.max_validation_examples)
        if args.validation_data
        else []
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = 0
    seen = 0
    skipped = 0
    optimizer.zero_grad(set_to_none=True)
    use_autocast = device.type == "cuda" and args.dtype == "bf16"
    if not use_autocast and args.dtype != "fp32":
        raise ValueError("BF16 training requires CUDA; use fp32 on CPU")
    from contextlib import nullcontext

    while steps < args.max_steps:
        rng.shuffle(train)
        for example in train:
            if steps >= args.max_steps:
                break
            mask_probability = rng.uniform(args.min_mask_probability, 1.0)
            try:
                student = build_la_sample(
                    tokenizer,
                    example.prompt,
                    example.answer,
                    mask_probability=mask_probability,
                    generator=generator,
                    canvas_tokens=args.canvas_tokens,
                    max_answer_tokens=args.max_answer_tokens,
                )
                if student.input_ids.numel() > model.backbone.config.max_position_embeddings:
                    skipped += 1
                    continue
                teacher_sample = (
                    build_la_sample(
                        teacher_tokenizer,
                        example.prompt,
                        example.answer,
                        mask_probability=mask_probability,
                        generator=torch.Generator().manual_seed(0),
                        canvas_tokens=args.canvas_tokens,
                        max_answer_tokens=args.max_answer_tokens,
                        use_native_template=True,
                    )
                    if teacher is not None
                    else None
                )
                if teacher_sample is not None:
                    # Use exactly the same corrupted Display as the student.
                    prefix_length = teacher_sample.input_ids.numel() - args.canvas_tokens
                    teacher_sample = LASample(
                        input_ids=torch.cat(
                            (
                                teacher_sample.input_ids[:prefix_length],
                                student.input_ids[-args.canvas_tokens :],
                            )
                        ),
                        targets=student.targets,
                        content_mask=student.content_mask,
                        masked_fraction=student.masked_fraction,
                    )
                    if (
                        teacher_sample.input_ids.numel()
                        > teacher.backbone.config.max_position_embeddings
                    ):
                        skipped += 1
                        continue
            except ValueError:
                skipped += 1
                continue
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
            )
            with autocast:
                logits = la_forward(model, student, device=device)
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = la_forward(teacher, teacher_sample, device=device)
                loss, ce, kl = la_objective(
                    logits,
                    student.targets.to(device),
                    teacher_logits=teacher_logits,
                    content_mask=student.content_mask.to(device),
                    distillation_weight=args.distillation_weight,
                    temperature=args.temperature,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite LA training loss")
            (loss / args.grad_accum).backward()
            seen += 1
            if seen % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            if steps == 1 or steps % args.log_every == 0:
                print(
                    f"la step={steps} examples={seen} skipped={skipped} "
                    f"loss={float(loss.detach()):.5f} ce={float(ce):.5f} "
                    f"kl={float(kl):.5f} masked={student.masked_fraction:.2f}",
                    flush=True,
                )
            if args.save_every and steps % args.save_every == 0 or steps == args.max_steps:
                manifest = dict(
                    version=1,
                    source_model=args.model,
                    native_teacher=args.native_teacher,
                    distillation_weight=args.distillation_weight,
                    steps=steps,
                    examples_seen=seen,
                    skipped=skipped,
                    canvas_tokens=args.canvas_tokens,
                    max_answer_tokens=args.max_answer_tokens,
                    format="raw-prompt-preallocated-display",
                )
                if validation:
                    manifest["validation_masked_ce"] = evaluate_la(
                        model,
                        tokenizer,
                        validation,
                        device=device,
                        canvas_tokens=args.canvas_tokens,
                        max_answer_tokens=args.max_answer_tokens,
                        seed=args.seed + 1,
                    )
                target_dir = out / f"la-step-{steps:06d}"
                save_la_backbone(model, tokenizer, target_dir, manifest)
                print("saved", target_dir, manifest.get("validation_masked_ce"), flush=True)
        if not seen or seen % args.grad_accum and steps == 0:
            raise ValueError("no usable LA training examples")
    print(f"LA complete steps={steps} output={out / f'la-step-{steps:06d}'}", flush=True)


def open_perfectblend_example(record: dict[str, Any]) -> StageCExample | None:
    conversation = record.get("conversations")
    if not isinstance(conversation, list):
        return None
    for position, message in enumerate(conversation):
        if not isinstance(message, dict) or message.get("from") != "human":
            continue
        prompt = message.get("value")
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        for response in conversation[position + 1 :]:
            if not isinstance(response, dict):
                continue
            if response.get("from") == "human":
                return None
            if response.get("from") == "gpt":
                answer = response.get("value")
                if isinstance(answer, str) and answer.strip():
                    return StageCExample(prompt.strip(), answer.strip())
                return None
    return None


def prepare_la_data(
    *,
    model: str,
    dataset: str,
    output_dir: str,
    max_scan: int,
    max_examples: int,
    max_prompt_tokens: int,
    min_answer_tokens: int,
    max_answer_tokens: int,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if (
        max_scan < 1
        or max_examples < 2
        or min_answer_tokens < 1
        or max_answer_tokens < min_answer_tokens
        or max_prompt_tokens < 1
        or not 0.0 < validation_fraction < 0.5
    ):
        raise ValueError("invalid LA data preparation limits")
    from collections import Counter

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    stream = load_dataset(dataset, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=1000)
    rng = random.Random(seed)
    sources: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    scanned = 0
    for record in stream:
        scanned += 1
        example = open_perfectblend_example(record)
        if example is None:
            rejected["unsupported"] += 1
        elif example.prompt in seen:
            rejected["duplicate"] += 1
        elif len(example.prompt) > 1500 or not 15 <= len(example.answer) <= 600:
            rejected["text_length"] += 1
        else:
            n_prompt = len(tokenizer.encode(example.prompt, add_special_tokens=True))
            n_answer = len(tokenizer.encode(example.answer, add_special_tokens=False))
            if (
                n_prompt > max_prompt_tokens
                or not min_answer_tokens <= n_answer <= max_answer_tokens
            ):
                rejected["token_budget"] += 1
            else:
                seen.add(example.prompt)
                source = str(record.get("source") or "unknown")
                sources[source] += 1
                selected.append(
                    {
                        "prompt": example.prompt,
                        "answer": example.answer,
                        "source": source,
                        "answer_tokens": n_answer,
                    }
                )
        if len(selected) >= max_examples or scanned >= max_scan:
            break
    if len(selected) < 2:
        raise ValueError("not enough qualifying instruction pairs for LA")
    rng.shuffle(selected)
    valid_count = max(1, round(len(selected) * validation_fraction))
    valid = selected[:valid_count]
    train = selected[valid_count:]
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, examples in (("train", train), ("validation", valid)):
        with (path / f"la-{name}.jsonl").open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    result = {
        "dataset": dataset,
        "seed": seed,
        "scanned": scanned,
        "train": len(train),
        "validation": len(valid),
        "sources": dict(sources),
        "rejected": dict(rejected),
        "min_answer_tokens": min_answer_tokens,
        "max_answer_tokens": max_answer_tokens,
    }
    (path / "la-data-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
