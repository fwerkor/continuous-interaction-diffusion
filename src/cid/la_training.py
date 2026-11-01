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
import torch.nn.functional as F

from cid.language_alignment import (
    LASample,
    build_la_sample,
    la_forward,
    load_la_backbone,
    save_la_backbone,
)


@dataclass(frozen=True)
class LARecord:
    prompt: str
    answer: str
    source: str
    language: str


def read_la_records(path: str, *, limit: int | None = None) -> list[LARecord]:
    out: list[LARecord] = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"LA dataset line {number} must contain a JSON object")
            prompt, answer = obj.get("prompt"), obj.get("answer")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(answer, str)
                or not answer.strip()
            ):
                raise ValueError(f"LA dataset line {number} has invalid prompt/answer")
            out.append(
                LARecord(
                    prompt.strip(),
                    answer.strip(),
                    str(obj.get("source", "")),
                    str(obj.get("language", "unknown")),
                )
            )
            if limit and len(out) >= limit:
                break
    if not out:
        raise ValueError(f"LA dataset {path} is empty")
    return out


def dataset_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canvas_bucket(answer_tokens: int, buckets: tuple[int, ...]) -> int:
    required = answer_tokens + 1
    for n in buckets:
        if n >= required:
            return n
    raise ValueError("answer too long for configured canvas buckets")


def token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eos_id: int,
    eos_weight: float,
) -> torch.Tensor:
    if not 0.0 < eos_weight <= 5.0:
        raise ValueError("EOS weight must be in (0, 5]")
    valid = labels.ne(-100)
    if not bool(valid.any()):
        raise ValueError("LA loss requires supervised answer tokens")
    per_token = F.cross_entropy(logits.float(), labels, ignore_index=-100, reduction="none")
    weights = torch.where(labels.eq(eos_id), eos_weight, 1.0).to(per_token.dtype)
    return (per_token * weights * valid).sum() / (weights * valid).sum().clamp_min(1)


def _make_example(
    tokenizer: Any,
    record: LARecord,
    *,
    generator: torch.Generator,
    probability: float,
    buckets: tuple[int, ...],
    max_answer_tokens: int,
    native: bool = False,
) -> LASample:
    length = len(tokenizer.encode(record.answer, add_special_tokens=False))
    width = canvas_bucket(length, buckets)
    return build_la_sample(
        tokenizer,
        record.prompt,
        record.answer,
        mask_probability=probability,
        generator=generator,
        canvas_tokens=width,
        max_answer_tokens=max_answer_tokens,
        use_native_template=native,
        supervise_eos_probability=0.7,
    )


def evaluate(
    model: Any,
    tokenizer: Any,
    records: list[LARecord],
    *,
    device: torch.device,
    buckets: tuple[int, ...],
    max_answer_tokens: int,
    eos_weight: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals: dict[str, list[float]] = {}
    with torch.inference_mode():
        for record in records:
            try:
                sample = _make_example(
                    tokenizer,
                    record,
                    generator=generator,
                    probability=0.85,
                    buckets=buckets,
                    max_answer_tokens=max_answer_tokens,
                )
            except ValueError:
                continue
            if sample.input_ids.numel() > model.backbone.config.max_position_embeddings:
                continue
            logits = la_forward(model, sample, device=device)
            loss = token_ce(
                logits,
                sample.targets.to(device),
                eos_id=tokenizer.eos_token_id,
                eos_weight=eos_weight,
            )
            v = float(loss)
            totals.setdefault("overall", []).append(v)
            totals.setdefault("language:" + record.language, []).append(v)
            totals.setdefault("source:" + record.source, []).append(v)
            totals.setdefault(
                "length:"
                + (
                    "short"
                    if len(sample.content_mask.nonzero()) <= 32
                    else "medium"
                    if len(sample.content_mask.nonzero()) <= 96
                    else "long"
                ),
                [],
            ).append(v)
    model.train()
    if not totals:
        raise ValueError("no LA validation samples fit model limits")
    return {key: sum(vals) / len(vals) for key, vals in totals.items()}


def _write_checkpoint(
    output: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    *,
    step: int,
    seen: int,
    skipped: int,
    order: list[int],
    position: int,
    rng: random.Random,
    generator: torch.Generator,
    sha: str,
    settings: dict[str, Any],
    metrics: dict[str, float],
    keep_last: int,
) -> Path:
    final = output / f"la-step-{step:07d}"
    temporary = output / f".la-step-{step:07d}.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        save_la_backbone(
            model,
            tokenizer,
            temporary,
            {
                **settings,
                "step": step,
                "seen": seen,
                "skipped": skipped,
                "validation": metrics,
                "dataset_sha256": sha,
            },
        )
        state = {
            "version": 2,
            "step": step,
            "seen": seen,
            "skipped": skipped,
            "optimizer": optimizer.state_dict(),
            "order": order,
            "position": position,
            "rng": rng.getstate(),
            "mask_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "dataset_sha256": sha,
            "settings": settings,
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state()
        torch.save(state, temporary / "training_state.pt")
        temporary.rename(final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    checkpoints = sorted(output.glob("la-step-*"))
    for old in checkpoints[:-keep_last]:
        shutil.rmtree(old)
    return final


def train_language_alignment_v2(args: Any) -> None:
    if args.max_steps < 1 or args.grad_accum < 1 or args.learning_rate <= 0:
        raise ValueError("invalid LA optimization settings")
    if not 0 < args.min_mask_probability <= 1 or not 0 < args.max_mask_probability <= 1:
        raise ValueError("invalid LA mask probabilities")
    if args.min_mask_probability > args.max_mask_probability:
        raise ValueError("minimum mask probability exceeds maximum")
    if args.save_every <= 0 or args.log_every <= 0 or args.keep_last < 1:
        raise ValueError("invalid LA checkpoint/log interval")
    if args.zh_repeat < 1.0 or args.eos_weight <= 0:
        raise ValueError("invalid LA language or EOS weighting")
    if args.warmup_steps < 0 or args.lr_decay_steps <= args.warmup_steps:
        raise ValueError("LA lr_decay_steps must exceed warmup_steps")
    if not 0.0 < args.min_lr_ratio <= 1.0:
        raise ValueError("minimum LA learning-rate ratio must lie in (0, 1]")
    buckets = tuple(int(x) for x in args.canvas_buckets.split(","))
    if (
        not buckets
        or sorted(set(buckets)) != list(buckets)
        or max(buckets) > args.max_answer_tokens
    ):
        raise ValueError("canvas buckets must be increasing, unique and <= answer budget")
    if args.distillation_weight and not args.native_teacher:
        raise ValueError("distillation requires --native-teacher")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if args.dtype == "bf16" and device.type != "cuda":
        raise ValueError("BF16 training requires a CUDA device")
    if args.dtype not in ("bf16", "fp32"):
        raise ValueError("unsupported training dtype")

    sha = dataset_sha256(args.data)
    records = read_la_records(args.data, limit=args.max_examples)
    validation = (
        read_la_records(args.validation_data, limit=args.max_validation_examples)
        if args.validation_data
        else []
    )
    if args.validation_data:
        overlap = set(x.prompt.lower().strip() for x in records) & set(
            x.prompt.lower().strip() for x in validation
        )
        if overlap:
            raise ValueError(f"training/validation prompt overlap: {len(overlap)}")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    resume = Path(args.resume) if args.resume else None
    model_path = str(resume) if resume else args.model
    model, tok = load_la_backbone(model_path, dtype=torch.float32)
    model.to(device).train()
    teacher = teacher_tok = None
    if args.native_teacher:
        teacher, teacher_tok = load_la_backbone(args.native_teacher, dtype=torch.float32)
        if tok.get_vocab() != teacher_tok.get_vocab():
            raise ValueError("teacher and student token IDs differ")
        teacher.to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "source_model": args.model,
        "learning_rate": args.learning_rate,
        "grad_accum": args.grad_accum,
        "max_answer_tokens": args.max_answer_tokens,
        "canvas_buckets": list(buckets),
        "eos_weight": args.eos_weight,
        "zh_repeat": args.zh_repeat,
        "native_teacher": args.native_teacher,
        "distillation_weight": args.distillation_weight,
        "temperature": args.temperature,
        "seed": args.seed,
        "min_mask_probability": args.min_mask_probability,
        "max_mask_probability": args.max_mask_probability,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "max_examples": args.max_examples,
        "warmup_steps": args.warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "min_lr_ratio": args.min_lr_ratio,
    }
    steps = seen = skipped = position = 0
    order: list[int] = []
    if resume:
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if state["dataset_sha256"] != sha:
            raise ValueError("resume dataset hash differs from initial training")
        if state["settings"] != settings:
            raise ValueError("resume hyperparameters differ from initial training")
        steps, seen, skipped = (state[x] for x in ("step", "seen", "skipped"))
        order, position = state["order"], state["position"]
        rng.setstate(state["rng"])
        gen.set_state(state["mask_rng"])
        torch.set_rng_state(state["torch_rng"])
        if "cuda_rng" in state and device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        optimizer.load_state_dict(state["optimizer"])
        for state_values in optimizer.state.values():
            for k, value in state_values.items():
                if isinstance(value, torch.Tensor):
                    state_values[k] = value.to(device)
        print("RESUMED", resume, "step", steps, "position", position, flush=True)
    if args.max_steps <= steps:
        raise ValueError("requested max_steps must exceed resumed step")
    if not order:
        order = list(range(len(records)))
        zh_indices = [i for i, r in enumerate(records) if r.language == "zh"]
        order += (
            rng.choices(zh_indices, k=round(len(zh_indices) * (args.zh_repeat - 1.0)))
            if zh_indices
            else []
        )
        rng.shuffle(order)

    def refresh_order() -> list[int]:
        rows = list(range(len(records)))
        zh_ids = [i for i, r in enumerate(records) if r.language == "zh"]
        if zh_ids and args.zh_repeat > 1:
            rows.extend(rng.choices(zh_ids, k=round(len(zh_ids) * (args.zh_repeat - 1))))
        rng.shuffle(rows)
        return rows

    optimizer.zero_grad(set_to_none=True)
    logpath = output / "training_metrics.jsonl"
    while steps < args.max_steps:
        if position >= len(order):
            order = refresh_order()
            position = 0
        record = records[order[position]]
        position += 1
        p = rng.uniform(args.min_mask_probability, args.max_mask_probability)
        try:
            sample = _make_example(
                tok,
                record,
                generator=gen,
                probability=p,
                buckets=buckets,
                max_answer_tokens=args.max_answer_tokens,
            )
            if sample.input_ids.numel() > model.backbone.config.max_position_embeddings:
                skipped += 1
                continue
            teacher_sample = None
            if teacher:
                teacher_sample = _make_example(
                    teacher_tok,
                    record,
                    generator=torch.Generator().manual_seed(0),
                    probability=p,
                    buckets=buckets,
                    max_answer_tokens=args.max_answer_tokens,
                    native=True,
                )
                prefix_len = teacher_sample.input_ids.numel() - sample.targets.numel()
                teacher_sample = LASample(
                    input_ids=torch.cat(
                        (
                            teacher_sample.input_ids[:prefix_len],
                            sample.input_ids[-sample.targets.numel() :],
                        )
                    ),
                    targets=sample.targets,
                    content_mask=sample.content_mask,
                    masked_fraction=sample.masked_fraction,
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
            torch.autocast("cuda", dtype=torch.bfloat16) if args.dtype == "bf16" else nullcontext()
        )
        with autocast:
            logits = la_forward(model, sample, device=device)
            ce = token_ce(
                logits,
                sample.targets.to(device),
                eos_id=tok.eos_token_id,
                eos_weight=args.eos_weight,
            )
            kl = ce.new_zeros(())
            if teacher:
                with torch.no_grad():
                    tlogits = la_forward(teacher, teacher_sample, device=device)
                content = sample.content_mask.to(device)
                if bool(content.any()):
                    scaled_s = logits[content].float() / args.temperature
                    scaled_t = tlogits[content].float() / args.temperature
                    kl = (
                        F.kl_div(
                            F.log_softmax(scaled_s, -1),
                            F.softmax(scaled_t, -1),
                            reduction="batchmean",
                        )
                        * args.temperature**2
                    )
            loss = ce + args.distillation_weight * kl
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("LA loss is NaN or infinite")
        (loss / args.grad_accum).backward()
        seen += 1
        if seen % args.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        next_step = steps + 1
        if next_step <= args.warmup_steps and args.warmup_steps:
            ratio = next_step / args.warmup_steps
        elif next_step >= args.lr_decay_steps:
            ratio = args.min_lr_ratio
        else:
            progress = (next_step - args.warmup_steps) / (args.lr_decay_steps - args.warmup_steps)
            ratio = (
                args.min_lr_ratio
                + (1.0 - args.min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2
            )
        lr = args.learning_rate * ratio
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        metric = {
            "step": steps,
            "lr": lr,
            "seen": seen,
            "skipped": skipped,
            "loss": float(loss.detach()),
            "ce": float(ce.detach()),
            "kl": float(kl.detach()),
            "mask": sample.masked_fraction,
            "width": sample.targets.numel(),
            "source": record.source,
            "language": record.language,
        }
        with logpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metric) + "\n")
        if steps == 1 or steps % args.log_every == 0:
            print("LA", json.dumps(metric), flush=True)
        if steps % args.save_every == 0 or steps == args.max_steps:
            metrics = (
                evaluate(
                    model,
                    tok,
                    validation,
                    device=device,
                    buckets=buckets,
                    max_answer_tokens=args.max_answer_tokens,
                    eos_weight=args.eos_weight,
                    seed=args.seed + 1,
                )
                if validation
                else {}
            )
            path = _write_checkpoint(
                output,
                model,
                tok,
                optimizer,
                step=steps,
                seen=seen,
                skipped=skipped,
                order=order,
                position=position,
                rng=rng,
                generator=gen,
                sha=sha,
                settings=settings,
                metrics=metrics,
                keep_last=args.keep_last,
            )
            print("CHECKPOINT", path, json.dumps(metrics), flush=True)
    print("COMPLETE", "steps", steps, "seen", seen, "skipped", skipped, flush=True)
