from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from cid.contracts import FreshnessDemand, InformationNeed, ModelContext, ModelUpdate
from cid.data import dump_jsonl, load_jsonl
from cid.dataset import dump_dataset_manifest, inspect_dataset
from cid.distill import (
    TeacherScheduleConfig,
    compile_teacher_plans,
    dump_teacher_plans,
    dump_teacher_requests,
    dump_teacher_reviews,
    dump_teacher_tasks,
    load_teacher_plans,
    load_teacher_tasks,
    review_teacher_plans,
    teacher_tasks_from_trajectories,
)
from cid.grounding import ObjectRef
from cid.metrics import summarize_runtime
from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry, StaticMappingSource
from cid.state import CognitiveField, CognitiveRole, DisplayCanvas
from cid.synthetic import SyntheticConfig, generate_synthetic


class DemoPolicy:
    def __init__(self, target_cell_id: str) -> None:
        self.target_cell_id = target_cell_id

    def step(self, context: ModelContext) -> ModelUpdate:
        time.sleep(0.02)
        need = InformationNeed(
            need_id="latency-spec",
            source_scores={"docs": 1.0},
            arguments={"key": "latency_ms"},
            confidence=1.0,
            freshness=FreshnessDemand.ONCE,
            target_cells=(ObjectRef.cell(self.target_cell_id),),
            target_display=(ObjectRef.display_span(0, 1),),
            promote_to_fact=True,
        )
        if context.percepts:
            value = int(context.percepts[0].observation.value)
            display = context.display.advance((value, 0, 0, 0))
            return ModelUpdate(
                thought=context.thought.advance(context.thought.cells),
                display=display,
                needs=(need,),
                converged=True,
            )
        return ModelUpdate(
            thought=context.thought.advance(context.thought.cells),
            display=context.display.advance(context.display.token_ids),
            needs=(need,),
        )


async def _run_demo() -> None:
    sources = SourceRegistry()
    sources.register(StaticMappingSource("docs", {"latency_ms": 37}, delay_s=0.03))
    runtime = CIDRuntime(sources, RuntimeConfig(max_steps=12, idle_yield_s=0.001))
    thought, need_cell_id = CognitiveField.empty(capacity=4, width=8).allocate(
        roles={CognitiveRole.INFORMATION_NEED: 1.0}
    )
    result = await runtime.run(
        DemoPolicy(need_cell_id),
        thought=thought,
        display=DisplayCanvas.masked(length=4, mask_token_id=-1),
    )
    metrics = summarize_runtime(result)
    print(f"converged={result.converged} steps={result.steps}")
    print(f"display={result.display.token_ids}")
    print(f"protected_facts={[(item.key, item.value) for item in result.facts]}")
    print(
        "interaction="
        f"external:{metrics.external_refreshes} "
        f"projections:{metrics.cognitive_projections} "
        f"model_steps_during_io:{metrics.model_steps_during_io}"
    )


def _generate_synthetic(args: argparse.Namespace) -> None:
    examples = generate_synthetic(
        SyntheticConfig(
            count_per_family=args.count_per_family,
            seed=args.seed,
            thought_capacity=args.thought_capacity,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(examples, output)
    print(f"wrote={len(examples)} path={output}")


def _prepare_distillation(args: argparse.Namespace) -> None:
    examples = load_jsonl(args.data)
    tasks = teacher_tasks_from_trajectories(examples)
    tasks_output = Path(args.tasks_output)
    requests_output = Path(args.requests_output)
    tasks_output.parent.mkdir(parents=True, exist_ok=True)
    requests_output.parent.mkdir(parents=True, exist_ok=True)
    dump_teacher_tasks(tasks, tasks_output)
    dump_teacher_requests(tasks, requests_output)
    print(
        f"tasks={len(tasks)} tasks_path={tasks_output} requests_path={requests_output}"
    )


def _compile_distillation(args: argparse.Namespace) -> None:
    tasks = load_teacher_tasks(args.tasks)
    plans = load_teacher_plans(args.plans)
    examples = compile_teacher_plans(
        tasks,
        plans,
        TeacherScheduleConfig(
            thought_capacity=args.thought_capacity,
            min_delay_steps=args.min_delay_steps,
            max_delay_steps=args.max_delay_steps,
            seed=args.seed,
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(examples, output)
    transition_count = sum(
        max(0, len({target.step for target in example.thought_targets}) - 1)
        for example in examples
    )
    print(
        f"compiled={len(examples)} transitions={transition_count} path={output}"
    )


def _review_distillation(args: argparse.Namespace) -> None:
    tasks = load_teacher_tasks(args.tasks)
    plans = load_teacher_plans(args.plans)
    reviews = review_teacher_plans(tasks, plans)
    accepted_ids = {review.task_id for review in reviews if review.accepted}
    accepted = tuple(plan for plan in plans if plan.task_id in accepted_ids)

    plans_output = Path(args.accepted_plans_output)
    report_output = Path(args.report_output)
    plans_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    dump_teacher_plans(accepted, plans_output)
    dump_teacher_reviews(reviews, report_output)
    rejected = len(reviews) - len(accepted)
    print(
        f"reviewed={len(reviews)} accepted={len(accepted)} rejected={rejected} "
        f"plans_path={plans_output} report_path={report_output}"
    )


def _dataset_manifest(args: argparse.Namespace) -> None:
    manifest = inspect_dataset(args.data)
    output = Path(args.output)
    dump_dataset_manifest(manifest, output)
    print(
        f"examples={manifest.examples} transitions={manifest.transitions} "
        f"sha256={manifest.sha256} path={output}"
    )


def _train_stage_a(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from cid.model import (
        ILLADA_8B_BASE,
        ILLADA_8B_BASE_REVISION,
        CIDTrainer,
        CIDTrainerConfig,
        ILLaDACIDAdapter,
        ILLaDACIDConfig,
        ILLaDATrajectoryTensorizer,
        shard_transitions,
        trajectory_transitions,
        wrap_stage_a_ddp,
    )

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        use_cuda = args.device != "cpu" and torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    else:
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        adapter_config = ILLaDACIDConfig(
            max_thought_slots=args.thought_capacity,
            max_display_tokens=args.max_display_tokens,
        )

        def load_adapter() -> ILLaDACIDAdapter:
            model = ILLaDACIDAdapter.from_pretrained(
                args.model,
                config=adapter_config,
                freeze_backbone=True,
                torch_dtype=dtype,
            )
            return model.to(device)

        adapter = None
        if distributed:
            for loading_rank in range(world_size):
                if rank == loading_rank:
                    adapter = load_adapter()
                dist.barrier()
        else:
            adapter = load_adapter()
        if adapter is None:
            raise RuntimeError("failed to load iLLaDA adapter on this training rank")
        if args.gradient_checkpointing:
            adapter.set_gradient_checkpointing(True)

        tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
        if args.model == ILLADA_8B_BASE:
            tokenizer_kwargs["revision"] = ILLADA_8B_BASE_REVISION
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
        tensorizer = ILLaDATrajectoryTensorizer(adapter, tokenizer)
        forward_model = (
            wrap_stage_a_ddp(
                adapter,
                device_ids=[local_rank] if str(device).startswith("cuda") else None,
            )
            if distributed
            else adapter
        )
        trainer = CIDTrainer(
            adapter,
            tensorizer,
            CIDTrainerConfig(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                micro_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                timestep_min=args.timestep_min,
                timestep_max=args.timestep_max,
                seed=args.seed,
            ),
            forward_model=forward_model,
        )
        if args.resume:
            trainer.load_checkpoint(args.resume)
        if distributed:
            trainer.reseed(args.seed + rank + trainer.state.transitions_seen * 104729)

        examples = load_jsonl(args.data)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        transitions = trajectory_transitions(examples)
        if not transitions:
            raise ValueError("training data contains no adjacent thought transitions")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainable = sum(
            parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
        )
        if rank == 0:
            effective_batch = (
                args.micro_batch_size * args.gradient_accumulation_steps * world_size
            )
            print(
                f"device={device} world_size={world_size} dtype={args.dtype} "
                f"examples={len(examples)} transitions={len(transitions)} "
                f"trainable_parameters={trainable} effective_batch={effective_batch}"
            )

        for epoch in range(1, args.epochs + 1):
            local_transitions = shard_transitions(
                transitions,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                epoch=epoch,
                shuffle=not args.no_shuffle,
            )
            report = trainer.train_transitions(local_transitions, epochs=1, shuffle=False)

            loss_sum = report.mean_loss * report.transitions
            transition_count = report.transitions
            if distributed:
                aggregate = torch.tensor(
                    [loss_sum, float(transition_count)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(aggregate)
                loss_sum = float(aggregate[0])
                transition_count = int(aggregate[1])
                dist.barrier()
            mean_loss = loss_sum / transition_count

            checkpoint = output_dir / f"stage-a-step-{trainer.state.optimizer_steps:08d}.pt"
            if rank == 0:
                trainer.save_checkpoint(checkpoint)
                print(
                    f"epoch={epoch} transitions={transition_count} "
                    f"optimizer_steps={report.optimizer_steps} mean_loss={mean_loss:.6f} "
                    f"checkpoint={checkpoint}"
                )
            if distributed:
                dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(prog="cid")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run the asynchronous static-source demo")
    synthetic = subparsers.add_parser(
        "generate-synthetic",
        help="generate deterministic CID mechanism-training trajectories",
    )
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--count-per-family", type=int, default=32)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--thought-capacity", type=int, default=8)
    prepare = subparsers.add_parser(
        "prepare-distillation",
        help="convert CID trajectories into timing-free teacher task/request JSONL",
    )
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--tasks-output", required=True)
    prepare.add_argument("--requests-output", required=True)
    compile_distillation = subparsers.add_parser(
        "compile-distillation",
        help="compile semantic teacher plans with independently randomized event schedules",
    )
    compile_distillation.add_argument("--tasks", required=True)
    compile_distillation.add_argument("--plans", required=True)
    compile_distillation.add_argument("--output", required=True)
    compile_distillation.add_argument("--thought-capacity", type=int, default=8)
    compile_distillation.add_argument("--min-delay-steps", type=int, default=1)
    compile_distillation.add_argument("--max-delay-steps", type=int, default=4)
    compile_distillation.add_argument("--seed", type=int, default=0)
    review = subparsers.add_parser(
        "review-distillation",
        help="quality-filter and deduplicate semantic teacher plans",
    )
    review.add_argument("--tasks", required=True)
    review.add_argument("--plans", required=True)
    review.add_argument("--accepted-plans-output", required=True)
    review.add_argument("--report-output", required=True)
    manifest = subparsers.add_parser(
        "dataset-manifest",
        help="write a deterministic manifest for a CID trajectory JSONL dataset",
    )
    manifest.add_argument("--data", required=True)
    manifest.add_argument("--output", required=True)
    train = subparsers.add_parser(
        "train",
        help="run Stage A CID adapter training with a frozen iLLaDA backbone",
    )
    train.add_argument("--data", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--model", default="GSAI-ML/iLLaDA-8B-Base")
    train.add_argument("--resume")
    train.add_argument("--device", default="auto")
    train.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--micro-batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--timestep-min", type=float, default=0.05)
    train.add_argument("--timestep-max", type=float, default=1.0)
    train.add_argument("--thought-capacity", type=int, default=8)
    train.add_argument("--max-display-tokens", type=int, default=1024)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--max-examples", type=int)
    train.add_argument("--no-shuffle", action="store_true")
    train.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.command == "demo":
        asyncio.run(_run_demo())
    elif args.command == "generate-synthetic":
        _generate_synthetic(args)
    elif args.command == "prepare-distillation":
        _prepare_distillation(args)
    elif args.command == "compile-distillation":
        _compile_distillation(args)
    elif args.command == "review-distillation":
        _review_distillation(args)
    elif args.command == "dataset-manifest":
        _dataset_manifest(args)
    elif args.command == "train":
        _train_stage_a(args)


if __name__ == "__main__":
    main()
