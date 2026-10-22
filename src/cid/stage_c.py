"""Optional Stage C: answer-only diffusion SFT on top of a frozen CID model.

Stage C learns no thought targets.  The frozen CID runtime may produce a T state
before each denoising example; only a Display-specific low-rank output residual
receives gradients.  The original adapter checkpoint is never modified.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

STAGE_C_FORMAT = "cid-stage-c-display-v1"
DEFAULT_DATASET = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v4"
DEFAULT_SPLITS = ("chat", "instruction_following_for_response_length_control")


class DisplayLowRankAdapter(nn.Module):
    """An output-only residual, not a modification to the tied LM head."""

    def __init__(
        self, hidden_size: int, vocab_size: int, rank: int = 32, alpha: float | None = None
    ):
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0 or rank <= 0:
            raise ValueError("Display LoRA dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        if not 0 < self.alpha < float("inf"):
            raise ValueError("Display LoRA alpha must be finite and positive")
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / (hidden_size**0.5))
        nn.init.zeros_(self.up.weight)
        self.enabled = True

    def forward(self, hidden: Tensor) -> Tensor:
        if not self.enabled:
            return hidden.new_zeros((*hidden.shape[:-1], self.vocab_size))
        # Train the small residual in fp32 even with a bf16 frozen backbone.
        value = self.up(self.down(hidden.float())) * (self.alpha / self.rank)
        return value.to(hidden.dtype)


def attach_display_adapter(adapter: nn.Module, *, rank: int = 32, alpha: float | None = None):
    if getattr(adapter, "stage_c_display", None) is not None:
        raise ValueError("a Stage C Display adapter is already attached")
    module = DisplayLowRankAdapter(adapter.d_model, adapter.vocab_size, rank, alpha)
    module.to(device=adapter.input_embeddings.weight.device)
    adapter.stage_c_display = module
    return module


def freeze_for_stage_c(adapter: nn.Module) -> DisplayLowRankAdapter:
    module = getattr(adapter, "stage_c_display", None)
    if not isinstance(module, DisplayLowRankAdapter):
        raise ValueError("attach Stage C Display adapter before freezing")
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    # Do not enable MoE router-auxiliary training in the frozen backbone.
    if hasattr(adapter, "_router_aux_loss_enabled"):
        adapter._router_aux_loss_enabled = False
    adapter.eval()
    module.train()
    return module


def save_display_adapter(
    path: str | Path, adapter: nn.Module, *, base_checkpoint: str = "", base_kind: str = ""
) -> None:
    module = getattr(adapter, "stage_c_display", None)
    if not isinstance(module, DisplayLowRankAdapter):
        raise ValueError("Stage C adapter is not attached")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": STAGE_C_FORMAT,
        "hidden_size": module.hidden_size,
        "vocab_size": module.vocab_size,
        "rank": module.rank,
        "alpha": module.alpha,
        "base_checkpoint": base_checkpoint,
        "base_kind": base_kind,
        "state_dict": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
    }
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_display_adapter(path: str | Path, adapter: nn.Module) -> DisplayLowRankAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != STAGE_C_FORMAT:
        raise ValueError("unrecognized Stage C adapter checkpoint")
    if (
        int(
            payload["hidden_size"],
        ),
        int(payload["vocab_size"]),
    ) != (adapter.d_model, adapter.vocab_size):
        raise ValueError("Stage C adapter geometry does not match CID backbone")
    module = attach_display_adapter(
        adapter, rank=int(payload["rank"]), alpha=float(payload["alpha"])
    )
    try:
        module.load_state_dict(payload["state_dict"], strict=True)
    except Exception:
        adapter.stage_c_display = None
        raise
    module.eval()
    return module


@dataclass(frozen=True, slots=True)
class StageCExample:
    prompt: str
    answer: str


def _content(message: Mapping[str, Any]) -> str:
    value = message.get("content", "")
    # Discard reasoning_content entirely: neither the prompt nor the label sees it.
    return value.strip() if isinstance(value, str) else ""


def examples_from_record(record: Mapping[str, Any]) -> tuple[StageCExample, ...]:
    messages = record.get("messages")
    if messages is None:
        prompt, answer = record.get("prompt"), record.get("answer", record.get("response"))
        return (
            (StageCExample(prompt.strip(), answer.strip()),)
            if isinstance(prompt, str)
            and prompt.strip()
            and isinstance(answer, str)
            and answer.strip()
            else ()
        )
    if not isinstance(messages, list):
        return ()
    metadata = record.get("metadata", {})
    mask = metadata.get("train_turns_mask") if isinstance(metadata, dict) else None
    if mask is not None and (not isinstance(mask, list) or len(mask) != len(messages)):
        raise ValueError("Nemotron train_turns_mask must align with messages")
    conversation: list[str] = []
    result: list[StageCExample] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = _content(message)
        if role not in {"system", "user", "assistant"} or not content:
            continue
        if (
            role == "assistant"
            and (mask is None or mask[index])
            and any(item.startswith("User:\n") for item in conversation)
        ):
            result.append(StageCExample("\n\n".join(conversation), content))
        conversation.append(f"{role.capitalize()}:\n{content}")
    return tuple(result)


def load_stage_c_examples(
    path: str | Path, *, max_examples: int | None = None
) -> list[StageCExample]:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    examples: list[StageCExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                examples.extend(examples_from_record(record))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid Stage C data at line {number}: {exc}") from exc
            if max_examples is not None and len(examples) >= max_examples:
                return examples[:max_examples]
    if not examples:
        raise ValueError("Stage C dataset contains no usable answers")
    return examples


def export_nemotron_sft(
    output: str | Path,
    *,
    dataset: str = DEFAULT_DATASET,
    splits: Iterable[str] = DEFAULT_SPLITS,
    max_per_split: int = 20000,
    seed: int = 42,
) -> dict[str, int]:
    """Stream and filter NVIDIA's public data; never ingest reasoning traces."""
    from datasets import load_dataset

    if max_per_split <= 0:
        raise ValueError("max_per_split must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with output.open("w", encoding="utf-8") as handle:
        for split in splits:
            stream = load_dataset(dataset, split=split, streaming=True)
            stream = stream.shuffle(seed=seed, buffer_size=10000)
            count = 0
            for record in stream:
                for example in examples_from_record(record):
                    handle.write(
                        json.dumps(
                            {"prompt": example.prompt, "answer": example.answer}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    count += 1
                    if count >= max_per_split:
                        break
                if count >= max_per_split:
                    break
            counts[split] = count
    return counts


def make_display_batch(
    tensorizer: Any,
    thought: Any,
    example: StageCExample,
    *,
    generator: torch.Generator,
    maximum_tokens: int = 512,
) -> tuple[Any, Tensor]:
    from cid.contracts import ModelContext
    from cid.state import DisplayCanvas, FactSnapshot

    adapter = tensorizer.adapter
    target = tensorizer.text_encoder.tokenize(example.answer, add_special_tokens=False)
    count = int(target.shape[1])
    if count < 1 or count + 1 > min(maximum_tokens, adapter.config.max_display_tokens):
        raise ValueError("Stage C answer exceeds Display training token budget")
    size = min(
        adapter.config.max_display_tokens, max(adapter.config.display_canvas_tokens, count + 1)
    )
    if (
        adapter.config.max_thought_slots
        + len(tensorizer.text_encoder.tokenize(example.prompt, add_special_tokens=True)[0])
        + size
        > adapter.max_position_embeddings
    ):
        raise ValueError("Stage C prompt and Display exceed backbone context length")
    mask_id = adapter.mask_token_id
    device = tensorizer.text_encoder.device
    tokens = torch.full((1, size), mask_id, dtype=torch.long, device=device)
    tokens[:, :count] = target.to(device)
    tokens[:, count] = adapter.eos_token_id
    eligible = torch.zeros_like(tokens, dtype=torch.bool)
    eligible[:, : count + 1] = True
    timestep = torch.rand((1,), device=device, generator=generator) * 0.95 + 0.05
    corrupted = tensorizer.scheduler.corrupt_display(
        tokens, timestep, eligible_mask=eligible, generator=generator
    )
    context = ModelContext(
        facts=FactSnapshot(0, {}),
        thought=thought,
        display=DisplayCanvas.initial_unresolved(
            length=size, mask_token_id=mask_id, eos_token_id=adapter.eos_token_id
        ),
        sources=(),
        percepts=(),
        step=0,
        prompt=example.prompt,
    )
    batch = tensorizer(context, generator=generator, display_noise_level=float(timestep.item()))
    batch.display_ids = corrupted.token_ids
    batch.display_noise = corrupted.noise.to(dtype=batch.display_noise.dtype)
    return batch, corrupted.labels


def frozen_thought_rollout(
    adapter: Any,
    tensorizer: Any,
    prompt: str,
    *,
    steps: int = 2,
    forward_model: Any | None = None,
) -> Any:
    """Run the existing frozen CID policy, without any teacher T labels or updates."""
    from cid.model.policy import ILLaDANeuralPolicy, ILLaDANeuralPolicyConfig
    from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry
    from cid.state import CognitiveField, DisplayCanvas

    initial = CognitiveField.empty(adapter.config.max_thought_slots, adapter.d_model)
    if steps == 0:
        return initial
    if steps < 0:
        raise ValueError("thought rollout steps cannot be negative")
    policy = ILLaDANeuralPolicy(
        adapter,
        tensorizer,
        config=ILLaDANeuralPolicyConfig(denoising_steps=max(steps, 2)),
        forward_model=forward_model,
    )
    runtime = CIDRuntime(SourceRegistry(), RuntimeConfig(max_steps=steps, max_wall_time_s=120))
    display = DisplayCanvas.initial_unresolved(
        length=adapter.config.display_canvas_tokens,
        mask_token_id=adapter.mask_token_id,
        eos_token_id=adapter.eos_token_id,
    )
    module = adapter.stage_c_display
    previous_enabled = module.enabled
    module.enabled = False
    try:
        result = asyncio.run(runtime.run(policy, thought=initial, display=display, prompt=prompt))
    finally:
        module.enabled = previous_enabled
    return result.thought


def display_sft_loss(
    adapter: Any, batch: Any, targets: Tensor, *, forward_model: Any | None = None
) -> Tensor:
    logits = (forward_model or adapter)(batch).display_logits
    # Only masked/changed tokens are trained, exactly as in CID's diffusion objective.
    if not bool((targets != -100).any()):
        raise ValueError("Stage C requires at least one supervised Display position")
    return F.cross_entropy(logits.float().transpose(1, 2), targets, ignore_index=-100)


def load_base_for_stage_c(
    checkpoint: str | Path,
    *,
    checkpoint_kind: str,
    model: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any]:
    """Restore the same CID weights and semantic encoder as benchmarking."""
    from cid.model import (
        ILLaDACIDConfig,
        load_cid_adapter_checkpoint,
        load_cid_adapter_from_pretrained,
        load_cid_model_from_pretrained,
        load_stage_b_semantic_encoder,
    )
    from cid.model.encoding import ILLaDATextEncoder
    from cid.model.loading import load_cid_tokenizer

    checkpoint = Path(checkpoint)
    if checkpoint_kind not in {"release", "stage-b"}:
        raise ValueError("Stage C base must be a Stage B or release checkpoint")
    if checkpoint_kind == "stage-b" and checkpoint.is_dir():
        raise ValueError(
            "sharded Stage B checkpoints must first be exported as unified CID releases"
        )
    if checkpoint_kind == "release":
        hf_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if hf_config.get("model_type") != "cid":
            raise ValueError("Stage C currently requires a unified CID release")
        model_obj = load_cid_model_from_pretrained(str(checkpoint), torch_dtype=dtype)
        adapter = model_obj.cid_adapter.to(device)
        tokenizer = load_cid_tokenizer(str(checkpoint))
        encoder = model_obj.make_text_encoder(tokenizer, device=device, embedding_device="cpu")
    else:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adapter_config = ILLaDACIDConfig(**raw["adapter_config"])
        has_semantic_snapshot = "semantic_embedding_snapshot" in raw
        pooling = raw.get("trainer_config", {}).get("semantic_pooling", "mean-v1")
        del raw
        adapter = load_cid_adapter_from_pretrained(
            model,
            config=adapter_config,
            freeze_backbone=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        tokenizer = load_cid_tokenizer(model)
        if has_semantic_snapshot:
            encoder = load_stage_b_semantic_encoder(
                adapter, tokenizer, checkpoint, device=device, embedding_device="cpu"
            )
        else:
            encoder = ILLaDATextEncoder.from_frozen_snapshot(
                adapter,
                tokenizer,
                device=device,
                dtype=dtype,
                embedding_device="cpu",
                pooling_mode=pooling,
            )
        load_cid_adapter_checkpoint(adapter, checkpoint, tokenizer=tokenizer)
    return adapter, tokenizer, encoder


def train_stage_c(args: Any) -> None:
    from cid.accelerator import (
        configure_torch_accelerator,
        resolve_torch_device_type,
        wrap_torch_autocast,
    )
    from cid.model.policy import ILLaDAContextTensorizer

    if args.epochs <= 0 or args.thought_steps < 0:
        raise ValueError("Stage C epochs must be positive and thought steps nonnegative")
    if args.learning_rate <= 0.0:
        raise ValueError("Stage C learning rate must be positive")
    device_type = resolve_torch_device_type(torch, args.device)
    configure_torch_accelerator(torch, device_type, 0)
    device = torch.device(device_type)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    adapter, tokenizer, encoder = load_base_for_stage_c(
        args.checkpoint,
        checkpoint_kind=args.checkpoint_kind,
        model=args.model,
        device=device,
        dtype=dtype,
    )
    torch.manual_seed(args.seed)
    module = attach_display_adapter(adapter, rank=args.rank, alpha=args.alpha)
    freeze_for_stage_c(adapter)
    tensorizer = ILLaDAContextTensorizer(adapter, tokenizer, text_encoder=encoder)
    forward_model = (
        adapter
        if dtype is torch.float32
        else wrap_torch_autocast(torch, adapter, device_type=device_type, dtype=dtype)
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    examples = load_stage_c_examples(args.data, max_examples=args.max_examples)
    rng = random.Random(args.seed)
    completed = 0
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        seen = 0
        for example in examples:
            try:
                answer_ids = encoder.tokenize(example.answer, add_special_tokens=False)
                prompt_ids = encoder.tokenize(example.prompt, add_special_tokens=True)
                answer_length = int(answer_ids.shape[1])
                canvas_length = max(adapter.config.display_canvas_tokens, answer_length + 1)
                if (
                    answer_length < 1
                    or answer_length + 1
                    > min(args.max_answer_tokens, adapter.config.max_display_tokens)
                    or adapter.config.max_thought_slots + int(prompt_ids.shape[1]) + canvas_length
                    > adapter.max_position_embeddings
                ):
                    continue
                thought = frozen_thought_rollout(
                    adapter,
                    tensorizer,
                    example.prompt,
                    steps=args.thought_steps,
                    forward_model=forward_model,
                )
                batch, labels = make_display_batch(
                    tensorizer,
                    thought,
                    example,
                    generator=generator,
                    maximum_tokens=args.max_answer_tokens,
                )
            except ValueError as exc:
                if "exceeds" in str(exc):
                    continue
                raise
            optimizer.zero_grad(set_to_none=True)
            loss = display_sft_loss(adapter, batch, labels, forward_model=forward_model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            completed += 1
            seen += 1
            if args.log_every and completed % args.log_every == 0:
                print(
                    f"stage=C epoch={epoch + 1} step={completed} loss={float(loss.detach()):.4f}",
                    flush=True,
                )
            if args.checkpoint_every and completed % args.checkpoint_every == 0:
                save_display_adapter(
                    Path(args.output_dir) / "stage-c-latest.pt",
                    adapter,
                    base_checkpoint=str(args.checkpoint),
                    base_kind=args.checkpoint_kind,
                )
        if seen == 0:
            raise ValueError("no Stage C samples fit configured context and Display limits")
        save_display_adapter(
            Path(args.output_dir) / f"stage-c-epoch-{epoch + 1:04d}.pt",
            adapter,
            base_checkpoint=str(args.checkpoint),
            base_kind=args.checkpoint_kind,
        )
        save_display_adapter(
            Path(args.output_dir) / "stage-c-latest.pt",
            adapter,
            base_checkpoint=str(args.checkpoint),
            base_kind=args.checkpoint_kind,
        )
    print(
        f"stage=C complete steps={completed} adapter={Path(args.output_dir) / 'stage-c-latest.pt'}"
    )
