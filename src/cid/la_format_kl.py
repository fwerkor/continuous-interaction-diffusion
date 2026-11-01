from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from cid.language_alignment import LABackbone, load_la_backbone, save_la_backbone

CID_ANSWER_TOKEN = "<|cid_answer|>"


@dataclass(frozen=True)
class PromptOnly:
    prompt: str
    source: str
    language: str


@dataclass(frozen=True)
class TeacherDraft:
    prompt: str
    text: str
    language: str


@dataclass(frozen=True)
class FormatPair:
    student_ids: Tensor
    teacher_ids: Tensor
    content_positions: Tensor
    eos_position: int | None
    display_length: int
    pseudo: bool


def read_prompts(path: str, limit: int | None = None) -> list[PromptOnly]:
    """Consume only prompts; even a JSONL answer column is never read."""
    records: list[PromptOnly] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"invalid prompt in line {i}")
            prompt = prompt.strip()
            key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            records.append(
                PromptOnly(prompt, str(row.get("source", "")), str(row.get("language", "")))
            )
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError("no prompts found")
    return records


def read_teacher_drafts(path: str, max_tokens: int, tokenizer: Any) -> list[TeacherDraft]:
    drafts: list[TeacherDraft] = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, text = row.get("prompt"), row.get("answer")
            if not isinstance(prompt, str) or not isinstance(text, str):
                raise ValueError("teacher drafts require prompt and teacher-generated text")
            if not prompt.strip() or not text.strip():
                continue
            # Never accept native role markup inside the CID Display target.
            if any(tag in text for tag in ("[Answer]", "[/Answer]", "[Question]", "[/Question]")):
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if 1 <= len(ids) + 2 <= max_tokens:
                drafts.append(
                    TeacherDraft(prompt.strip(), text.strip(), str(row.get("language", "")))
                )
    if not drafts:
        raise ValueError("no usable native-teacher generations")
    return drafts


def configure_cid_student(model: LABackbone, tokenizer: Any, *, resume: bool) -> int:
    """Use a reserved embedding row without changing pretrained vocabulary geometry."""
    old_vocab = len(tokenizer)
    existing = tokenizer.convert_tokens_to_ids(CID_ANSWER_TOKEN)
    if resume:
        if existing == tokenizer.unk_token_id or existing >= old_vocab:
            raise ValueError("resumed student checkpoint lacks CID answer boundary")
        return int(existing)
    if CID_ANSWER_TOKEN in tokenizer.get_vocab():
        raise ValueError("base model already has the CID answer special token")
    added = tokenizer.add_special_tokens({"additional_special_tokens": [CID_ANSWER_TOKEN]})
    if added != 1:
        raise ValueError("expected one new CID boundary token")
    token_id = int(tokenizer.convert_tokens_to_ids(CID_ANSWER_TOKEN))
    embeddings = model.backbone.get_input_embeddings()
    if token_id >= embeddings.num_embeddings:
        # Future models may not reserve embeddings; keep the operation explicit.
        model.backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        embeddings = model.backbone.get_input_embeddings()
    native_prefix = tokenizer.encode("[Answer]\n", add_special_tokens=False)
    if not native_prefix:
        raise ValueError("cannot initialize CID boundary from existing native prefix")
    with torch.no_grad():
        embeddings.weight[token_id].copy_(
            embeddings.weight[native_prefix].detach().float().mean(0).to(embeddings.weight.dtype)
        )
    return token_id


def build_pair(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    record: PromptOnly | TeacherDraft,
    *,
    canvas_width: int,
    student_answer_id: int,
    generator: torch.Generator,
    mask_probability: float,
    max_content_positions: int,
) -> FormatPair:
    if canvas_width < 4 or not (0.0 < mask_probability <= 1.0):
        raise ValueError("invalid format distillation canvas or mask probability")
    teacher_prefix = teacher_tokenizer.apply_chat_template(
        [{"role": "user", "content": record.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    teacher_prompt = teacher_tokenizer.encode(teacher_prefix, add_special_tokens=True)
    student_prompt = [
        *student_tokenizer.encode(record.prompt, add_special_tokens=True),
        student_answer_id,
    ]
    mask = int(teacher_tokenizer.mask_token_id)
    if student_tokenizer.mask_token_id != mask:
        raise ValueError("teacher and student mask token must match")
    canvas = [mask] * canvas_width
    eos_position = None
    if isinstance(record, TeacherDraft):
        answer = teacher_tokenizer.encode(record.text, add_special_tokens=False)
        if not answer or len(answer) + 2 > canvas_width:
            raise ValueError("teacher-generated draft does not fit canvas")
        draws = torch.rand((len(answer),), generator=generator)
        hidden = draws < mask_probability
        if not bool(hidden.any()):
            hidden[int(draws.argmin())] = True
        for i, token in enumerate(answer):
            if not bool(hidden[i]):
                canvas[i] = token
        positions = hidden.nonzero(as_tuple=True)[0]
        eos_position = len(answer)
    else:
        # Early answer positions are the useful all-MASK distribution, not
        # arbitrary far-tail positions beyond the native answer's stopping point.
        positions = torch.arange(min(canvas_width, max_content_positions))
    if positions.numel() > max_content_positions:
        selection = torch.randperm(positions.numel(), generator=generator)[:max_content_positions]
        positions = positions[selection].sort().values
    ids = torch.tensor(canvas, dtype=torch.long)
    return FormatPair(
        student_ids=torch.tensor([*student_prompt, *canvas], dtype=torch.long),
        teacher_ids=torch.tensor([*teacher_prompt, *canvas], dtype=torch.long),
        content_positions=positions,
        eos_position=eos_position,
        display_length=ids.numel(),
        pseudo=isinstance(record, TeacherDraft),
    )


def pure_format_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    positions: Tensor,
    *,
    temperature: float,
    eos_position: int | None,
    eos_id: int,
    termination_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Teacher soft KL plus optional teacher-generated termination-event KL.

    The EOS term is KL(delta_teacher_stop || student), not dataset-answer CE.
    No dataset answer is involved in either term.
    """
    if temperature <= 0 or termination_weight < 0:
        raise ValueError("invalid KL temperature or stop weight")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher/student vocabulary dimensions differ")
    if positions.numel() == 0:
        raise ValueError("KL needs at least one masked content position")
    pos = positions.to(student_logits.device)
    t = F.softmax(teacher_logits.detach().index_select(0, pos).float() / temperature, -1)
    s = F.log_softmax(student_logits.index_select(0, pos).float() / temperature, -1)
    content = F.kl_div(s, t, reduction="batchmean") * (temperature * temperature)
    terminal = content.new_zeros(())
    if eos_position is not None and termination_weight:
        if not 0 <= eos_position < student_logits.shape[0]:
            raise ValueError("EOS position outside Display canvas")
        terminal = -F.log_softmax(student_logits[eos_position].float(), -1)[eos_id]
    return content + termination_weight * terminal, content.detach(), terminal.detach()


def _checkpoint(
    dest: Path,
    student: LABackbone,
    tokenizer: Any,
    optimizer: Any,
    *,
    step: int,
    seen: int,
    generator: torch.Generator,
    rng: random.Random,
    settings: dict[str, Any],
    validation: dict[str, float],
    keep_last: int,
) -> Path:
    target = dest / f"la-step-{step:07d}"
    staging = dest / f".la-step-{step:07d}.incomplete"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        save_la_backbone(
            student,
            tokenizer,
            staging,
            {
                **settings,
                "objective": "prompt_only_teacher_kl",
                "step": step,
                "seen": seen,
                "validation": validation,
            },
        )
        state = {
            "version": 1,
            "settings": settings,
            "step": step,
            "seen": seen,
            "optimizer": optimizer.state_dict(),
            "rng": rng.getstate(),
            "mask_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state()
        torch.save(state, staging / "training_state.pt")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for old in sorted(dest.glob("la-step-*"))[:-keep_last]:
        shutil.rmtree(old)
    return target


def train_prompt_only_format_kl(args: Any) -> None:
    if args.max_steps <= 0 or args.grad_accum <= 0 or args.learning_rate <= 0:
        raise ValueError("invalid optimizer configuration")
    if args.max_steps < args.warmup_steps or args.decay_steps <= args.warmup_steps:
        raise ValueError("invalid learning-rate schedule")
    if args.save_every < 1 or args.log_every < 1 or args.keep_last < 1:
        raise ValueError("invalid checkpoint interval")
    if not 0 <= args.pseudo_probability <= 1:
        raise ValueError("invalid teacher-generation sampling probability")
    if args.pseudo_probability and not args.teacher_generations:
        raise ValueError("teacher generation file required for pseudo states")
    if not 0 < args.mask_min <= args.mask_max <= 1:
        raise ValueError("invalid partial-mask range")
    buckets = tuple(int(s) for s in args.canvas_buckets.split(","))
    if not buckets or sorted(set(buckets)) != list(buckets) or min(buckets) < 4:
        raise ValueError("invalid canvas buckets")
    if args.max_prompt_tokens < 1:
        raise ValueError("invalid prompt token budget")
    if args.max_content_positions < 1 or args.termination_weight < 0:
        raise ValueError("invalid content or stop configuration")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if args.dtype == "bf16" and device.type != "cuda":
        raise ValueError("BF16 training requires CUDA")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    prompts = read_prompts(args.data, limit=args.max_examples)
    heldout = (
        read_prompts(args.validation_data, limit=args.max_validation_examples)
        if args.validation_data
        else []
    )
    if heldout and {r.prompt for r in prompts} & {r.prompt for r in heldout}:
        raise ValueError("training and validation prompts overlap")
    resume = Path(args.resume) if args.resume else None
    student, student_tok = load_la_backbone(str(resume or args.model), dtype=torch.float32)
    answer_id = configure_cid_student(student, student_tok, resume=resume is not None)
    teacher, teacher_tok = load_la_backbone(args.teacher, dtype=torch.float32)
    if (
        student.backbone.get_output_embeddings().weight.shape
        != teacher.backbone.get_output_embeddings().weight.shape
    ):
        raise ValueError("student and teacher logits have different dimensions")
    if (
        student_tok.mask_token_id != teacher_tok.mask_token_id
        or student_tok.eos_token_id != teacher_tok.eos_token_id
    ):
        raise ValueError("student and teacher MASK/EOS mismatch")
    drafts = (
        read_teacher_drafts(args.teacher_generations, max(buckets), teacher_tok)
        if args.teacher_generations
        else []
    )
    student.to(device).train()
    teacher.to(device).eval().requires_grad_(False)
    opt = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    settings = {
        "model": args.model,
        "teacher": args.teacher,
        "data_sha256": file_sha256(args.data),
        "validation_sha256": file_sha256(args.validation_data) if args.validation_data else None,
        "teacher_generations_sha256": file_sha256(args.teacher_generations)
        if args.teacher_generations
        else None,
        "learning_rate": args.learning_rate,
        "decay_steps": args.decay_steps,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "grad_accum": args.grad_accum,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "canvas_buckets": buckets,
        "max_content_positions": args.max_content_positions,
        "max_prompt_tokens": args.max_prompt_tokens,
        "pseudo_probability": args.pseudo_probability,
        "mask_min": args.mask_min,
        "mask_max": args.mask_max,
        "termination_weight": args.termination_weight,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "max_validation_examples": args.max_validation_examples,
        "prompt_only": True,
        "cid_answer_token_id": answer_id,
    }
    step = seen = 0
    if resume:
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("resume hyperparameters or dataset differ")
        step, seen = state["step"], state["seen"]
        rng.setstate(state["rng"])
        generator.set_state(state["mask_rng"])
        torch.set_rng_state(state["torch_rng"])
        if "cuda_rng" in state and device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        opt.load_state_dict(state["optimizer"])
        for entry in opt.state.values():
            for key, value in entry.items():
                if isinstance(value, Tensor):
                    entry[key] = value.to(device)
        print("RESUMED", str(resume), "step", step, flush=True)
    if step >= args.max_steps:
        raise ValueError("requested max steps must exceed resumed checkpoint")
    train_stat = {"all_mask": 0, "teacher_partial": 0}

    def run_pair(
        record: PromptOnly | TeacherDraft, *, mask_probability: float, width: int
    ) -> tuple[Tensor, Tensor, Tensor, FormatPair]:
        pair = build_pair(
            student_tok,
            teacher_tok,
            record,
            canvas_width=width,
            student_answer_id=answer_id,
            generator=generator,
            mask_probability=mask_probability,
            max_content_positions=args.max_content_positions,
        )
        if (
            max(
                pair.student_ids.numel() - width,
                pair.teacher_ids.numel() - width,
            )
            > args.max_prompt_tokens
        ):
            raise ValueError("prompt exceeds training budget")
        if (
            pair.student_ids.numel() > student.backbone.config.max_position_embeddings
            or pair.teacher_ids.numel() > teacher.backbone.config.max_position_embeddings
        ):
            raise ValueError("prompt exceeds model context")
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16) if args.dtype == "bf16" else nullcontext()
        )
        with autocast:
            student_logits = student(pair.student_ids.unsqueeze(0).to(device))[0, -width:]
            with torch.no_grad():
                teacher_logits = teacher(pair.teacher_ids.unsqueeze(0).to(device))[0, -width:]
            loss, content, terminal = pure_format_kl(
                student_logits,
                teacher_logits,
                pair.content_positions,
                temperature=args.temperature,
                eos_position=pair.eos_position,
                eos_id=student_tok.eos_token_id,
                termination_weight=args.termination_weight,
            )
        return loss, content, terminal, pair

    print(
        "PURE_KL_START",
        json.dumps(
            {
                "prompts": len(prompts),
                "teacher_drafts": len(drafts),
                "teacher_answer_dataset_column_used": False,
                "cid_answer_id": answer_id,
                "max_steps": args.max_steps,
                "gpu": str(device),
                "initial_step": step,
            }
        ),
        flush=True,
    )
    opt.zero_grad(set_to_none=True)
    logpath = dest / "training_metrics.jsonl"
    while step < args.max_steps:
        is_pseudo = bool(drafts) and rng.random() < args.pseudo_probability
        record: PromptOnly | TeacherDraft = rng.choice(drafts if is_pseudo else prompts)
        if is_pseudo:
            required = len(teacher_tok.encode(record.text, add_special_tokens=False)) + 2
            possible = [b for b in buckets if b >= required]
            if not possible:
                continue
            width = rng.choice(possible)
            p = rng.uniform(args.mask_min, args.mask_max)
        else:
            width = rng.choice(buckets)
            p = 1.0
        try:
            loss, content, terminal, pair = run_pair(record, mask_probability=p, width=width)
        except ValueError as exc:
            # Invalid user prompt/overlong teacher-generated drafts are skipped,
            # but model/protocol errors must fail fast.
            if "exceeds model context" in str(exc) or "exceeds training budget" in str(exc):
                continue
            raise
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("pure KL loss became nonfinite")
        (loss / args.grad_accum).backward()
        train_stat["teacher_partial" if is_pseudo else "all_mask"] += 1
        seen += 1
        if seen % args.grad_accum:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        next_step = step + 1
        if next_step <= args.warmup_steps and args.warmup_steps:
            ratio = next_step / args.warmup_steps
        elif next_step >= args.decay_steps:
            ratio = args.min_lr_ratio
        else:
            progress = (next_step - args.warmup_steps) / (args.decay_steps - args.warmup_steps)
            ratio = (
                args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
            )
        lr = args.learning_rate * ratio
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        step = next_step
        metric = {
            "step": step,
            "seen": seen,
            "lr": lr,
            "loss": float(loss.detach()),
            "kl": float(content),
            "teacher_stop_kl": float(terminal),
            "grad_norm": float(grad_norm),
            "canvas": width,
            "mode": "teacher_partial" if is_pseudo else "all_mask",
            "source": "native_teacher" if is_pseudo else record.source,
            "language": record.language,
        }
        with logpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metric) + "\n")
        if step == 1 or step % args.log_every == 0:
            print("PURE_KL", json.dumps(metric), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            student.eval()
            values = []
            with torch.inference_mode():
                for row in heldout:
                    width = buckets[0]
                    try:
                        _, k, _, _ = run_pair(row, mask_probability=1.0, width=width)
                    except ValueError as exc:
                        if "exceeds model context" in str(exc) or "exceeds training budget" in str(
                            exc
                        ):
                            continue
                        raise
                    values.append(float(k))
            student.train()
            validation = {"heldout_full_mask_kl": sum(values) / len(values)} if values else {}
            path = _checkpoint(
                dest,
                student,
                student_tok,
                opt,
                step=step,
                seen=seen,
                generator=generator,
                rng=rng,
                settings=settings,
                validation=validation,
                keep_last=args.keep_last,
            )
            print("PURE_KL_CHECKPOINT", str(path), json.dumps(validation), flush=True)
    print("PURE_KL_COMPLETE", json.dumps({"steps": step, "seen": seen, **train_stat}), flush=True)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
