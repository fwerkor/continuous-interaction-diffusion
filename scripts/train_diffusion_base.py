from __future__ import annotations

import argparse
import json
import os
import queue
import random
import shutil
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from huggingface_hub import HfFileSystem
from torch import Tensor
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    MixedPrecision,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

from cid.model.diffusion_base import (
    ARDiffusionBase,
    PackedCorpusSampler,
    fsdp_accumulation_context,
    masked_diffusion_loss,
    stage0_learning_rate,
)


class RemoteTokenSource:
    def __init__(
        self,
        spec: dict[str, object],
        *,
        tokenizer: Any,
        sequence_length: int,
        rank: int,
        world_size: int,
        seed: int,
        evaluation: bool,
    ) -> None:
        self.name = str(spec["name"])
        self.weight = float(spec["weight"])
        self.column = str(spec["column"])
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        eos = tokenizer.eos_token_id
        if eos is None:
            raise RuntimeError("streaming corpus tokenizer must define eos_token_id")
        self.eos_token_id = int(eos)
        repo = str(spec["repo"])
        revision = str(spec["revision"])
        pattern = str(spec["pattern"])
        fs = HfFileSystem()
        paths = sorted(fs.glob(f"datasets/{repo}@{revision}/{pattern}"))
        if len(paths) <= world_size:
            raise RuntimeError(f"streaming source {self.name!r} has too few parquet shards")
        held_out = paths[-world_size:]
        if evaluation:
            self.paths = [held_out[rank]]
        else:
            train_paths = paths[:-world_size]
            random.Random(seed).shuffle(train_paths)
            self.paths = train_paths[rank::world_size]
        if not self.paths:
            raise RuntimeError(f"streaming source {self.name!r} assigned no parquet shards")
        self._path_index = 0
        self._documents = self._document_iterator()
        self._buffer: list[int] = []
        self._offset = 0

    def _document_iterator(self):
        while True:
            path = self.paths[self._path_index % len(self.paths)]
            self._path_index += 1
            with fsspec.open("hf://" + path, "rb") as stream:
                parquet = pq.ParquetFile(stream)
                for batch in parquet.iter_batches(batch_size=64, columns=[self.column]):
                    texts = []
                    for value in batch.column(0):
                        text = value.as_py()
                        if text and text.strip():
                            texts.append(text)
                    if not texts:
                        continue
                    encoded = self.tokenizer(
                        texts,
                        add_special_tokens=False,
                        padding=False,
                        truncation=False,
                    )["input_ids"]
                    for token_ids in encoded:
                        if token_ids:
                            yield token_ids

    def next_sequence(self) -> np.ndarray:
        needed = self.sequence_length
        while len(self._buffer) - self._offset < needed:
            token_ids = next(self._documents)
            self._buffer.extend(token_ids)
            self._buffer.append(self.eos_token_id)
        end = self._offset + needed
        result = np.asarray(self._buffer[self._offset : end], dtype=np.int64)
        self._offset = end
        if self._offset >= 16384:
            del self._buffer[: self._offset]
            self._offset = 0
        return result


class StreamingCorpusSampler:
    def __init__(
        self,
        manifest: dict[str, object],
        *,
        tokenizer: Any,
        rank: int,
        world_size: int,
        seed: int,
        evaluation: bool = False,
        prefetch_sequences: int = 32,
    ) -> None:
        self.sequence_length = int(manifest["sequence_length"])
        source_specs = list(manifest["sources"])
        self.sources = tuple(
            RemoteTokenSource(
                spec,
                tokenizer=tokenizer,
                sequence_length=self.sequence_length,
                rank=rank,
                world_size=world_size,
                seed=seed + index * 104729,
                evaluation=evaluation,
            )
            for index, spec in enumerate(source_specs)
        )
        weights = np.asarray([source.weight for source in self.sources], dtype=np.float64)
        self.weights = weights / weights.sum()
        self.rng = np.random.default_rng(seed)
        self._queue: queue.Queue[np.ndarray | BaseException] = queue.Queue(
            maxsize=prefetch_sequences
        )
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        tokenizer: Any,
        rank: int,
        world_size: int,
        seed: int,
        evaluation: bool = False,
    ) -> StreamingCorpusSampler:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            manifest,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            seed=seed,
            evaluation=evaluation,
        )

    def _produce(self) -> None:
        try:
            while True:
                source_index = int(self.rng.choice(len(self.sources), p=self.weights))
                self._queue.put(self.sources[source_index].next_sequence())
        except BaseException as exc:
            self._queue.put(exc)

    def sample(self, batch_size: int) -> Tensor:
        rows = []
        for _ in range(batch_size):
            item = self._queue.get()
            if isinstance(item, BaseException):
                raise RuntimeError("streaming corpus producer failed") from item
            rows.append(item)
        return torch.from_numpy(np.stack(rows, axis=0))


def build_sampler(
    manifest_path: Path,
    *,
    tokenizer: Any,
    rank: int,
    world_size: int,
    seed: int,
    evaluation: bool = False,
):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") == "cid-diffusion-stream-v1":
        return StreamingCorpusSampler(
            manifest,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            seed=seed,
            evaluation=evaluation,
        )
    return PackedCorpusSampler.from_manifest(manifest_path, seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B-Base")
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--target-global-batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-mask-ratio", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--resume")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fsdp-no-sync-accumulation",
        dest="no_sync_accumulation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--aggressive-prefetch", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--export-hf", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def wrap_fsdp(
    model: ARDiffusionBase,
    *,
    device: torch.device,
    cpu_offload: bool,
    aggressive_prefetch: bool,
) -> FSDP:
    layers = model.backbone.get_decoder().layers
    layer_class = type(layers[0])
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_class},
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,
        backward_prefetch=(
            BackwardPrefetch.BACKWARD_PRE if aggressive_prefetch else BackwardPrefetch.BACKWARD_POST
        ),
        forward_prefetch=aggressive_prefetch,
        limit_all_gathers=not aggressive_prefetch,
        use_orig_params=True,
    )


def sharded_state_context(model: FSDP):
    return FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    )


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    *,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    data_manifest: dict[str, object],
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step:07d}"
    if dist.get_rank() == 0:
        checkpoint.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    with sharded_state_context(model):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=checkpoint / "distributed",
    )
    if dist.get_rank() == 0:
        metadata = {
            "format": "cid-diffusion-base-stage0-v1",
            "step": step,
            "base_model": args.model,
            "training": {
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "lr_schedule": args.lr_schedule,
                "warmup_ratio": args.warmup_ratio,
                "micro_batch_size": args.micro_batch_size,
                "target_global_batch_size": args.target_global_batch_size,
                "min_mask_ratio": args.min_mask_ratio,
            },
            "data_manifest": data_manifest,
        }
        (checkpoint / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    dist.barrier()
    if dist.get_rank() == 0 and args.keep_checkpoints > 0:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        for stale in checkpoints[: -args.keep_checkpoints]:
            shutil.rmtree(stale)
    dist.barrier()
    return checkpoint


def load_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
) -> int:
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    with sharded_state_context(model):
        model_state = model.state_dict()
    state: dict[str, object] = {"model": model_state}
    dcp.load(state, checkpoint_id=checkpoint / "distributed")
    with sharded_state_context(model):
        model.load_state_dict(state["model"])
    # DCP needs the current sharded model layout to rebuild optimizer state.
    import torch.distributed.checkpoint.optimizer as dcp_optimizer

    optimizer_state = dcp_optimizer.load_sharded_optimizer_state_dict(
        model_state,
        optimizer_key="optimizer",
        storage_reader=dcp.FileSystemReader(checkpoint / "distributed"),
    )["optimizer"]
    flattened = FSDP.optim_state_dict_to_load(model, optimizer, optimizer_state)
    optimizer.load_state_dict(flattened)
    return int(metadata["step"])


def evaluate(
    model: FSDP,
    sampler,
    *,
    batches: int,
    batch_size: int,
    device: torch.device,
    mask_token_id: int,
    min_mask_ratio: float,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = torch.zeros((), device=device, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(batches):
            clean = sampler.sample(batch_size).to(device=device, non_blocking=True)
            loss, _, _ = masked_diffusion_loss(
                model,
                clean,
                mask_token_id=mask_token_id,
                min_mask_ratio=min_mask_ratio,
                generator=generator,
            )
            total += loss.detach().float()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    model.train()
    return float(total.cpu()) / (batches * dist.get_world_size())


def export_hf(
    model: FSDP,
    tokenizer,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    data_manifest: dict[str, object],
    completed_steps: int,
) -> None:
    export_dir = output_dir / "hf"
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state = model.state_dict()
    if dist.get_rank() != 0:
        return
    export_dir.mkdir(parents=True, exist_ok=True)
    prefix = "backbone."
    backbone_state = {
        key[len(prefix) :]: value for key, value in full_state.items() if key.startswith(prefix)
    }
    if not backbone_state:
        raise RuntimeError("full FSDP state did not contain backbone parameters")
    backbone = model.module.backbone
    backbone.config.mask_token_id = int(model.module.mask_token_id)
    backbone.config.use_cache = False
    backbone.config.cid_diffusion_base = True
    backbone.config.diffusion_objective = "masked-diffusion"
    backbone.config.diffusion_attention = "bidirectional"
    backbone.config.base_model_name_or_path = args.model
    backbone.save_pretrained(
        export_dir,
        state_dict=backbone_state,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(export_dir)
    diffusion_config = {
        "format": "cid-diffusion-base-v1",
        "base_model": args.model,
        "objective": "LLaDA-style masked diffusion",
        "attention": "bidirectional",
        "mask_token": tokenizer.mask_token,
        "mask_token_id": int(model.module.mask_token_id),
        "sequence_length": int(data_manifest["sequence_length"]),
        "completed_steps": completed_steps,
        "global_batch_size": args.target_global_batch_size,
        "tokens_seen": (
            completed_steps * args.target_global_batch_size * int(data_manifest["sequence_length"])
        ),
        "training_corpus": data_manifest,
    }
    (export_dir / "diffusion_config.json").write_text(
        json.dumps(diffusion_config, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.micro_batch_size <= 0 or args.target_global_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 0 currently requires CUDA")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.target_global_batch_size % (world_size * args.micro_batch_size) != 0:
        raise ValueError("target global batch must be divisible by world_size * micro_batch_size")
    accumulation_steps = args.target_global_batch_size // (world_size * args.micro_batch_size)
    if accumulation_steps > 1 and not args.no_sync_accumulation and args.cpu_offload:
        raise ValueError("FSDP CPU offload accumulation requires --no-sync-accumulation")

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest_path = Path(args.data_manifest)
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    sequence_length = int(data_manifest["sequence_length"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = build_sampler(
        data_manifest_path,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        seed=args.seed + rank * 1_000_003,
        evaluation=False,
    )
    eval_sampler = build_sampler(
        data_manifest_path,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        seed=args.seed + 99_000_001 + rank,
        evaluation=True,
    )

    # Serialize model loading to reduce storage pressure from three simultaneous HF reads.
    backbone = None
    for loading_rank in range(world_size):
        if rank == loading_rank:
            backbone = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        dist.barrier()
    assert backbone is not None
    backbone.config.use_cache = False
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = ARDiffusionBase(backbone, tokenizer)
    fsdp = wrap_fsdp(
        model,
        device=device,
        cpu_offload=args.cpu_offload,
        aggressive_prefetch=args.aggressive_prefetch,
    )
    optimizer = torch.optim.AdamW(
        fsdp.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(fsdp, optimizer, Path(args.resume))

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    metrics_path = output_dir / f"train_metrics.rank-{rank:04d}.jsonl"
    start_time = time.monotonic()
    last_log_time = start_time
    interval_loss = torch.zeros((), device=device, dtype=torch.float32)
    interval_mask = torch.zeros((), device=device, dtype=torch.float32)
    interval_ratio = torch.zeros((), device=device, dtype=torch.float32)
    interval_steps = 0
    fsdp.train()

    for step in range(start_step + 1, args.steps + 1):
        lr = stage0_learning_rate(
            step,
            total_steps=args.steps,
            peak_lr=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_learning_rate_ratio,
            schedule=args.lr_schedule,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device, dtype=torch.float32)
        step_mask = torch.zeros((), device=device, dtype=torch.float32)
        step_ratio = torch.zeros((), device=device, dtype=torch.float32)
        for micro_step in range(accumulation_steps):
            clean = sampler.sample(args.micro_batch_size).to(device=device, non_blocking=True)
            synchronize = micro_step == accumulation_steps - 1 or not args.no_sync_accumulation
            context = fsdp_accumulation_context(fsdp, synchronize=synchronize)
            with context:
                loss, masked_fraction, mean_ratio = masked_diffusion_loss(
                    fsdp,
                    clean,
                    mask_token_id=model.mask_token_id,
                    min_mask_ratio=args.min_mask_ratio,
                    generator=generator,
                )
                (loss / accumulation_steps).backward()
            step_loss += loss.detach().float()
            step_mask += masked_fraction.float()
            step_ratio += mean_ratio.float()
        grad_norm = fsdp.clip_grad_norm_(args.max_grad_norm)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite Stage 0 gradient norm at step {step}")
        optimizer.step()

        step_loss /= accumulation_steps
        step_mask /= accumulation_steps
        step_ratio /= accumulation_steps
        interval_loss += step_loss
        interval_mask += step_mask
        interval_ratio += step_ratio
        interval_steps += 1

        if step % args.log_every == 0 or step == 1:
            packed = torch.stack((interval_loss, interval_mask, interval_ratio))
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            packed /= world_size * interval_steps
            now = time.monotonic()
            elapsed = now - start_time
            interval_elapsed = now - last_log_time
            tokens_per_step = args.target_global_batch_size * sequence_length
            record = {
                "step": step,
                "loss": float(packed[0].cpu()),
                "masked_fraction": float(packed[1].cpu()),
                "mean_mask_ratio": float(packed[2].cpu()),
                "learning_rate": lr,
                "grad_norm": float(grad_norm.detach().float().cpu()),
                "elapsed_seconds": elapsed,
                "seconds_per_step": interval_elapsed / interval_steps,
                "tokens_per_second": tokens_per_step * interval_steps / interval_elapsed,
                "tokens_seen": step * tokens_per_step,
                "sequence_length": sequence_length,
                "global_batch_size": args.target_global_batch_size,
                "world_size": world_size,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if rank == 0:
                print(json.dumps(record), flush=True)
            interval_loss.zero_()
            interval_mask.zero_()
            interval_ratio.zero_()
            interval_steps = 0
            last_log_time = now

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_loss = evaluate(
                fsdp,
                eval_sampler,
                batches=args.eval_batches,
                batch_size=args.micro_batch_size,
                device=device,
                mask_token_id=model.mask_token_id,
                min_mask_ratio=args.min_mask_ratio,
                seed=args.seed + 700_000_000 + step + rank,
            )
            if rank == 0:
                with (output_dir / "eval_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "loss": eval_loss}) + "\n")
                print(json.dumps({"eval_step": step, "eval_loss": eval_loss}), flush=True)

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(
                fsdp,
                optimizer,
                output_dir=output_dir,
                step=step,
                args=args,
                data_manifest=data_manifest,
            )

    if args.checkpoint_every > 0 and args.steps % args.checkpoint_every != 0:
        save_checkpoint(
            fsdp,
            optimizer,
            output_dir=output_dir,
            step=args.steps,
            args=args,
            data_manifest=data_manifest,
        )

    if args.export_hf:
        export_hf(
            fsdp,
            tokenizer,
            output_dir=output_dir,
            args=args,
            data_manifest=data_manifest,
            completed_steps=args.steps,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
