from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from cid.computational_training import (
    ComputationalTrainingConfig,
    build_computational_training,
)
from cid.contracts import FreshnessDemand, InformationNeed, ModelContext, ModelUpdate
from cid.correction_training import CorrectionTrainingConfig, build_correction_training
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
from cid.evaluation import summarize_evaluations
from cid.grounding import ObjectRef
from cid.metrics import summarize_runtime
from cid.multilingual_training import MultilingualTrainingConfig, build_multilingual_training
from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry, StaticMappingSource
from cid.self_identity_training import (
    SelfIdentityTrainingConfig,
    build_self_identity_training,
)
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


def _build_computational_training(args: argparse.Namespace) -> None:
    manifest = build_computational_training(
        tasks_output=args.tasks_output,
        requests_output=args.requests_output,
        causal_jobs_output=args.causal_jobs_output,
        manifest_output=args.manifest_output,
        config=ComputationalTrainingConfig(
            count_per_family=args.count_per_family,
            seed=args.seed,
        ),
    )
    print(
        f"tasks={manifest['tasks']} families={len(manifest['family_counts'])} "
        f"tasks_path={manifest['tasks_path']} causal_jobs={manifest['causal_jobs']}"
    )


def _build_correction_training(args: argparse.Namespace) -> None:
    manifest = build_correction_training(
        tasks_output=args.tasks_output,
        requests_output=args.requests_output,
        causal_jobs_output=args.causal_jobs_output,
        manifest_output=args.manifest_output,
        config=CorrectionTrainingConfig(
            count_per_family=args.count_per_family,
            seed=args.seed,
        ),
    )
    print(
        f"tasks={manifest['tasks']} families={len(manifest['family_counts'])} "
        f"tasks_path={manifest['tasks_path']} causal_jobs={manifest['causal_jobs']}"
    )


def _build_symbolic_training(args: argparse.Namespace) -> None:
    from cid.symbolic_training import SymbolicTrainingConfig, build_symbolic_training

    manifest = build_symbolic_training(
        tasks_output=args.tasks_output,
        requests_output=args.requests_output,
        causal_jobs_output=args.causal_jobs_output,
        manifest_output=args.manifest_output,
        config=SymbolicTrainingConfig(
            count_per_family=args.count_per_family,
            seed=args.seed,
        ),
    )
    print(
        f"tasks={manifest['tasks']} families={len(manifest['family_counts'])} "
        f"tasks_path={manifest['tasks_path']} causal_jobs={manifest['causal_jobs']}"
    )


def _build_self_identity_training(args: argparse.Namespace) -> None:
    manifest = build_self_identity_training(
        contract_path=args.contract,
        tasks_output=args.tasks_output,
        plans_output=args.plans_output,
        reviews_output=args.reviews_output,
        trajectories_output=args.trajectories_output,
        trajectory_manifest_output=args.trajectory_manifest_output,
        reference_manifest_output=args.reference_manifest_output,
        config=SelfIdentityTrainingConfig(
            count_per_family=args.count_per_family,
            seed=args.seed,
            variants_per_task=args.variants_per_task,
            thought_capacity=args.thought_capacity,
        ),
    )
    print(
        f"tasks={manifest['semantic_tasks']} accepted={manifest['accepted_plans']} "
        f"trajectories={manifest['compiled_trajectories']} "
        f"transitions={manifest['compiled_transitions']}"
    )


def _build_multilingual_training(args: argparse.Namespace) -> None:
    manifest = build_multilingual_training(
        args.output_dir,
        config=MultilingualTrainingConfig(
            zh_tasks=args.zh_tasks,
            en_zh_tasks=args.en_zh_tasks,
            ja_tasks=args.ja_tasks,
            es_tasks=args.es_tasks,
            schedule_variants=args.schedule_variants,
            seed=args.seed,
        ),
    )
    print(
        f"tasks={manifest['semantic_tasks']} trajectories={manifest['compiled_trajectories']} "
        f"cross_lingual={manifest['cross_lingual_tasks']} output={args.output_dir}"
    )


def _prepare_distillation(args: argparse.Namespace) -> None:
    from cid.causal_distill import dump_causal_teacher_jobs

    examples = load_jsonl(args.data)
    tasks = teacher_tasks_from_trajectories(examples)
    tasks_output = Path(args.tasks_output)
    requests_output = Path(args.requests_output)
    tasks_output.parent.mkdir(parents=True, exist_ok=True)
    requests_output.parent.mkdir(parents=True, exist_ok=True)
    dump_teacher_tasks(tasks, tasks_output)
    dump_teacher_requests(tasks, requests_output)
    if args.causal_jobs_output:
        causal_output = Path(args.causal_jobs_output)
        causal_output.parent.mkdir(parents=True, exist_ok=True)
        dump_causal_teacher_jobs(tasks, causal_output)
    print(
        f"tasks={len(tasks)} tasks_path={tasks_output} requests_path={requests_output} "
        f"causal_jobs={args.causal_jobs_output}"
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
            variants_per_task=args.variants_per_task,
            seed=args.seed,
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(examples, output)
    transition_count = sum(
        max(0, len({target.step for target in example.thought_targets}) - 1) for example in examples
    )
    print(f"compiled={len(examples)} transitions={transition_count} path={output}")


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


def _materialize_trajectory_mixture(args: argparse.Namespace) -> None:
    from cid.trajectory_mixture import materialize_trajectory_mixture

    manifest = materialize_trajectory_mixture(args.spec, args.output, args.manifest_output)
    print(
        f"examples={manifest['examples']} transitions={manifest['transitions']} "
        f"sha256={manifest['sha256']} path={args.output} manifest={args.manifest_output}"
    )


def _build_public_task_pool(args: argparse.Namespace) -> None:
    from cid.public_tasks import build_public_task_pool

    manifest = build_public_task_pool(args.registry, args.output, args.manifest_output)
    print(
        f"tasks={manifest['tasks']} sha256={manifest['sha256']} "
        f"output={args.output} manifest={args.manifest_output}"
    )


def _prepare_public_distillation(args: argparse.Namespace) -> None:
    from cid.public_training import PublicTrainingConfig, prepare_public_distillation

    manifest = prepare_public_distillation(
        args.data,
        args.tasks_output,
        args.requests_output,
        args.manifest_output,
        PublicTrainingConfig(
            split=args.split,
            seed=args.seed,
            unnecessary_tool_fraction=args.unnecessary_tool_fraction,
        ),
        causal_jobs_output=args.causal_jobs_output,
    )
    print(
        f"tasks={manifest['tasks']} split={manifest['split']} "
        f"modes={manifest['mode_counts']} tasks_path={args.tasks_output} "
        f"requests_path={args.requests_output} causal_jobs={args.causal_jobs_output} "
        f"manifest={args.manifest_output}"
    )


def _teacher_wave_export(args: argparse.Namespace) -> None:
    from cid.teacher_wave import export_teacher_wave

    report = export_teacher_wave(
        args.jobs,
        args.state,
        args.output,
        max_requests=args.max_requests,
    )
    print(
        f"jobs={report['jobs']} complete_tasks={report['complete_tasks']} "
        f"exported={report['exported_requests']} path={args.output}"
    )


def _teacher_wave_import(args: argparse.Namespace) -> None:
    from cid.teacher_wave import import_teacher_wave

    report = import_teacher_wave(
        args.jobs,
        args.requests,
        args.responses,
        args.state,
        rejects_path=args.rejects_output,
    )
    print(
        f"imported={report['imported']} unchanged={report['unchanged']} "
        f"rejected={report['rejected']} state_records={report['state_records']} "
        f"state={args.state}"
    )


def _teacher_wave_finalize(args: argparse.Namespace) -> None:
    from cid.teacher_wave import finalize_teacher_wave

    tasks = load_teacher_tasks(args.tasks)
    plans = finalize_teacher_wave(tasks, args.jobs, args.state, args.output)
    print(f"plans={len(plans)} path={args.output}")


def _teacher_wave_status(args: argparse.Namespace) -> None:
    from cid.teacher_wave import teacher_wave_status

    report = teacher_wave_status(args.jobs, args.state)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _teacher_agent_checkout(args: argparse.Namespace) -> None:
    from cid.teacher_agent import checkout_teacher_agent_batch

    report = checkout_teacher_agent_batch(
        args.jobs,
        args.state,
        args.workspace,
        max_requests=args.max_requests,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _teacher_agent_commit(args: argparse.Namespace) -> None:
    from cid.teacher_agent import commit_teacher_agent_batch

    report = commit_teacher_agent_batch(args.workspace)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _benchmark(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from cid.model import (
        ILLADA_8B_BASE,
        ILLADA_8B_BASE_REVISION,
        ILLaDACIDAdapter,
        ILLaDACIDConfig,
        load_cid_adapter_checkpoint,
        load_stage_b_model_checkpoint,
        wrap_stage_b_fsdp,
    )
    from cid.model.benchmark import run_neural_benchmark_case
    from cid.model.encoding import ILLaDATextEncoder

    checkpoint = Path(args.checkpoint)
    stage_b = args.checkpoint_kind == "stage-b"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = stage_b
    if stage_b:
        if world_size < 2:
            raise RuntimeError("Stage B benchmark must run under multi-GPU torchrun")
        if not torch.cuda.is_available():
            raise RuntimeError("Stage B benchmark requires CUDA GPUs")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        adapter_config = ILLaDACIDConfig(**metadata["adapter_config"])
    else:
        if world_size > 1:
            raise RuntimeError("Stage A benchmark is single-process; omit torchrun")
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adapter_config = ILLaDACIDConfig(**raw["adapter_config"])

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
    if args.model == ILLADA_8B_BASE:
        tokenizer_kwargs["revision"] = ILLADA_8B_BASE_REVISION
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)

    try:

        def load_adapter() -> ILLaDACIDAdapter:
            return ILLaDACIDAdapter.from_pretrained(
                args.model,
                config=adapter_config,
                freeze_backbone=True,
                torch_dtype=torch.float32 if stage_b else dtype,
                low_cpu_mem_usage=True,
            ).to(device)

        adapter = None
        if stage_b:
            for loading_rank in range(world_size):
                if rank == loading_rank:
                    adapter = load_adapter()
                dist.barrier()
        else:
            adapter = load_adapter()
        if adapter is None:
            raise RuntimeError("failed to load benchmark iLLaDA adapter")

        if stage_b:
            text_encoder = ILLaDATextEncoder.from_frozen_snapshot(
                adapter,
                tokenizer,
                device=device,
                dtype=dtype,
            )
            adapter.set_backbone_trainable(True)
            forward_model = wrap_stage_b_fsdp(
                adapter,
                device_id=device,
                compute_dtype=dtype,
            )
            load_stage_b_model_checkpoint(forward_model, adapter, checkpoint)
        else:
            load_cid_adapter_checkpoint(adapter, checkpoint)
            text_encoder = ILLaDATextEncoder(adapter, tokenizer)
            forward_model = adapter

        adapter.eval()
        forward_model.eval()
        examples = load_jsonl(args.data)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        if not examples:
            raise ValueError("benchmark dataset is empty")

        async def run_cases():
            results = []
            for index, example in enumerate(examples, start=1):
                result = await run_neural_benchmark_case(
                    adapter,
                    tokenizer,
                    example,
                    text_encoder=text_encoder,
                    forward_model=forward_model,
                    seed_teacher_state=args.seed_teacher_state,
                    denoising_steps=args.denoising_steps,
                    max_steps=args.max_steps,
                )
                results.append(result)
                if distributed:
                    dist.barrier()
                if rank == 0 and args.progress_every and index % args.progress_every == 0:
                    print(f"benchmarked={index}/{len(examples)}")
            return tuple(results)

        results = asyncio.run(run_cases())
        if rank == 0:
            output = Path(args.output)
            summary_output = Path(args.summary_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            summary_output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as handle:
                for result in results:
                    handle.write(
                        json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
            summary = summarize_evaluations(tuple(result.evaluation for result in results))
            payload = {
                "checkpoint": str(checkpoint),
                "checkpoint_kind": args.checkpoint_kind,
                "model": args.model,
                "seed_teacher_state": args.seed_teacher_state,
                "metrics": asdict(summary),
            }
            summary_output.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"tasks={summary.tasks} exact={summary.exact_display_accuracy:.4f} "
                f"converged={summary.convergence_rate:.4f} "
                f"coverage={summary.observation_coverage:.4f} output={output}"
            )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


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
        shard_rollout_windows,
        trajectory_rollout_windows,
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
                rollout_horizon=args.rollout_horizon,
                teacher_forcing_epochs=args.teacher_forcing_epochs,
                rollout_ramp_epochs=args.rollout_ramp_epochs,
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
        windows = trajectory_rollout_windows(
            examples,
            max_horizon=args.rollout_horizon,
        )
        if not windows:
            raise ValueError("training data contains no adjacent thought transitions")
        transition_count_total = sum(len(window.source_steps) for window in windows)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainable = sum(
            parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
        )
        if rank == 0:
            effective_batch = args.micro_batch_size * args.gradient_accumulation_steps * world_size
            print(
                f"device={device} world_size={world_size} dtype={args.dtype} "
                f"examples={len(examples)} transitions={transition_count_total} "
                f"trainable_parameters={trainable} effective_batch={effective_batch}"
            )

        first_epoch = trainer.state.epochs_completed + 1
        for epoch in range(first_epoch, first_epoch + args.epochs):
            local_windows = shard_rollout_windows(
                windows,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                epoch=epoch,
                shuffle=not args.no_shuffle,
            )
            rollout_probability = trainer.rollout_probability()
            report = trainer.train_rollout_windows(local_windows, epochs=1, shuffle=False)

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
                    f"rollout_probability={rollout_probability:.3f} checkpoint={checkpoint}"
                )
            if distributed:
                dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train_stage_b(args: argparse.Namespace) -> None:
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
        load_cid_adapter_checkpoint,
        load_stage_b_checkpoint,
        save_stage_b_checkpoint,
        shard_rollout_windows,
        trajectory_rollout_windows,
        wrap_stage_b_fsdp,
    )
    from cid.model.encoding import ILLaDATextEncoder

    if args.resume and args.init_cid_checkpoint:
        raise ValueError("--resume and --init-cid-checkpoint are mutually exclusive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("Stage B full-parameter training must run under multi-GPU torchrun")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage B full-parameter training requires CUDA GPUs")

    compute_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        dataset_manifest = inspect_dataset(args.data)
        if dataset_manifest.thought_capacity_required > args.thought_capacity:
            raise ValueError(
                "training data requires a larger TCT capacity: "
                f"{dataset_manifest.thought_capacity_required} > {args.thought_capacity}"
            )
        tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
        if args.model == ILLADA_8B_BASE:
            tokenizer_kwargs["revision"] = ILLADA_8B_BASE_REVISION
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
        adapter_config = ILLaDACIDConfig(
            max_thought_slots=args.thought_capacity,
            max_display_tokens=args.max_display_tokens,
        )

        def load_adapter() -> tuple[ILLaDACIDAdapter, ILLaDATextEncoder]:
            model = ILLaDACIDAdapter.from_pretrained(
                args.model,
                config=adapter_config,
                freeze_backbone=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)
            if args.init_cid_checkpoint:
                load_cid_adapter_checkpoint(model, args.init_cid_checkpoint)
            snapshot = ILLaDATextEncoder.from_frozen_snapshot(
                model,
                tokenizer,
                device=device,
                dtype=compute_dtype,
            )
            model.set_backbone_trainable(True)
            if args.gradient_checkpointing:
                model.set_gradient_checkpointing(True)
            return model, snapshot

        adapter = None
        text_encoder = None
        for loading_rank in range(world_size):
            if rank == loading_rank:
                adapter, text_encoder = load_adapter()
            dist.barrier()
        if adapter is None or text_encoder is None:
            raise RuntimeError("failed to load Stage B iLLaDA model on this training rank")

        fsdp_model = wrap_stage_b_fsdp(
            adapter,
            device_id=device,
            compute_dtype=compute_dtype,
        )
        optimizer = torch.optim.AdamW(
            fsdp_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        tensorizer = ILLaDATrajectoryTensorizer(
            adapter,
            tokenizer,
            text_encoder=text_encoder,
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
                rollout_horizon=args.rollout_horizon,
                teacher_forcing_epochs=args.teacher_forcing_epochs,
                rollout_ramp_epochs=args.rollout_ramp_epochs,
                seed=args.seed,
            ),
            optimizer=optimizer,
            forward_model=fsdp_model,
            gradient_clipper=fsdp_model.clip_grad_norm_,
        )
        if args.resume:
            load_stage_b_checkpoint(
                fsdp_model,
                optimizer,
                trainer,
                args.resume,
                expected_dataset_sha256=dataset_manifest.sha256,
            )
        else:
            trainer.reseed(args.seed + rank)

        examples = load_jsonl(args.data)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        windows = trajectory_rollout_windows(
            examples,
            max_horizon=args.rollout_horizon,
        )
        if not windows:
            raise ValueError("training data contains no adjacent thought transitions")
        transition_count_total = sum(len(window.source_steps) for window in windows)
        output_dir = Path(args.output_dir)
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            effective_batch = args.micro_batch_size * args.gradient_accumulation_steps * world_size
            print(
                f"stage=B device={device} world_size={world_size} dtype={args.dtype} "
                f"examples={len(examples)} transitions={transition_count_total} "
                f"effective_batch={effective_batch}"
            )
        dist.barrier()

        first_epoch = trainer.state.epochs_completed + 1
        for epoch in range(first_epoch, first_epoch + args.epochs):
            local_windows = shard_rollout_windows(
                windows,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                epoch=epoch,
                shuffle=not args.no_shuffle,
            )
            rollout_probability = trainer.rollout_probability()
            report = trainer.train_rollout_windows(local_windows, epochs=1, shuffle=False)
            aggregate = torch.tensor(
                [report.mean_loss * report.transitions, float(report.transitions)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(aggregate)
            mean_loss = float(aggregate[0] / aggregate[1])
            transition_count = int(aggregate[1])

            checkpoint = output_dir / f"stage-b-step-{trainer.state.optimizer_steps:08d}"
            save_stage_b_checkpoint(
                fsdp_model,
                optimizer,
                trainer,
                checkpoint,
                dataset_sha256=dataset_manifest.sha256,
            )
            if rank == 0:
                print(
                    f"epoch={epoch} transitions={transition_count} "
                    f"optimizer_steps={report.optimizer_steps} mean_loss={mean_loss:.6f} "
                    f"rollout_probability={rollout_probability:.3f} checkpoint={checkpoint}"
                )
    finally:
        if dist.is_initialized():
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
    computational = subparsers.add_parser(
        "build-computational-training",
        help="build generated calculator/Python/lookup semantic teacher tasks",
    )
    computational.add_argument(
        "--tasks-output", default="data/generated/computational-teacher-tasks-v1.jsonl"
    )
    computational.add_argument(
        "--requests-output", default="data/generated/computational-teacher-requests-v1.jsonl"
    )
    computational.add_argument(
        "--causal-jobs-output", default="data/generated/computational-teacher-causal-v1.jsonl"
    )
    computational.add_argument(
        "--manifest-output", default="data/generated/computational-teacher-v1.manifest.json"
    )
    computational.add_argument("--count-per-family", type=int, default=1200)
    computational.add_argument("--seed", type=int, default=20260812)
    correction = subparsers.add_parser(
        "build-correction-training",
        help="build speculative-error/local-correction semantic teacher tasks",
    )
    correction.add_argument(
        "--tasks-output", default="data/generated/correction-teacher-tasks-v1.jsonl"
    )
    correction.add_argument(
        "--requests-output", default="data/generated/correction-teacher-requests-v1.jsonl"
    )
    correction.add_argument(
        "--causal-jobs-output", default="data/generated/correction-teacher-causal-v1.jsonl"
    )
    correction.add_argument(
        "--manifest-output", default="data/generated/correction-teacher-v1.manifest.json"
    )
    correction.add_argument("--count-per-family", type=int, default=1000)
    correction.add_argument("--seed", type=int, default=20260812)
    symbolic = subparsers.add_parser(
        "build-symbolic-training",
        help="build generated symbolic algebra/calculus semantic teacher tasks",
    )
    symbolic.add_argument(
        "--tasks-output", default="data/generated/symbolic-teacher-tasks-v1.jsonl"
    )
    symbolic.add_argument(
        "--requests-output", default="data/generated/symbolic-teacher-requests-v1.jsonl"
    )
    symbolic.add_argument(
        "--causal-jobs-output", default="data/generated/symbolic-teacher-causal-v1.jsonl"
    )
    symbolic.add_argument(
        "--manifest-output", default="data/generated/symbolic-teacher-v1.manifest.json"
    )
    symbolic.add_argument("--count-per-family", type=int, default=1200)
    symbolic.add_argument("--seed", type=int, default=20260812)
    self_identity = subparsers.add_parser(
        "build-self-identity-training",
        help="build deterministic CID model-identity and architecture self-knowledge training data",
    )
    self_identity.add_argument("--contract", default="data/cid-self-identity-v1.contract.json")
    self_identity.add_argument(
        "--tasks-output", default="data/generated/self-identity-teacher-tasks-v1.jsonl"
    )
    self_identity.add_argument(
        "--plans-output", default="data/generated/self-identity-teacher-plans-v1.accepted.jsonl"
    )
    self_identity.add_argument(
        "--reviews-output", default="data/generated/self-identity-review-v1.jsonl"
    )
    self_identity.add_argument(
        "--trajectories-output", default="data/generated/self-identity-trajectories-v1.jsonl"
    )
    self_identity.add_argument(
        "--trajectory-manifest-output",
        default="data/generated/self-identity-trajectories-v1.manifest.json",
    )
    self_identity.add_argument(
        "--reference-manifest-output",
        default="data/generated/self-identity-reference-manifest-v1.json",
    )
    self_identity.add_argument("--count-per-family", type=int, default=80)
    self_identity.add_argument("--variants-per-task", type=int, default=2)
    self_identity.add_argument("--thought-capacity", type=int, default=8)
    self_identity.add_argument("--seed", type=int, default=20260813)
    multilingual = subparsers.add_parser(
        "build-multilingual-training",
        help="build a small multilingual and cross-lingual CID trajectory component",
    )
    multilingual.add_argument(
        "--output-dir", default="data/generated/multilingual-v1"
    )
    multilingual.add_argument("--zh-tasks", type=int, default=450)
    multilingual.add_argument("--en-zh-tasks", type=int, default=300)
    multilingual.add_argument("--ja-tasks", type=int, default=225)
    multilingual.add_argument("--es-tasks", type=int, default=225)
    multilingual.add_argument("--schedule-variants", type=int, default=2)
    multilingual.add_argument("--seed", type=int, default=20260813)
    prepare = subparsers.add_parser(
        "prepare-distillation",
        help="convert CID trajectories into timing-free teacher task/request JSONL",
    )
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--tasks-output", required=True)
    prepare.add_argument("--requests-output", required=True)
    prepare.add_argument("--causal-jobs-output")
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
    compile_distillation.add_argument("--variants-per-task", type=int, default=1)
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
    trajectory_mixture = subparsers.add_parser(
        "materialize-trajectory-mixture",
        help="verify and concatenate pinned CID trajectory components into one training JSONL",
    )
    trajectory_mixture.add_argument("--spec", required=True)
    trajectory_mixture.add_argument("--output", required=True)
    trajectory_mixture.add_argument("--manifest-output", required=True)
    public_pool = subparsers.add_parser(
        "build-public-task-pool",
        help="build the pinned public semantic-task pool from registered training splits",
    )
    public_pool.add_argument("--registry", default="data/public-datasets.json")
    public_pool.add_argument("--output", default="data/generated/public-task-pool-v1.jsonl")
    public_pool.add_argument(
        "--manifest-output",
        default="data/generated/public-task-pool-v1.manifest.json",
    )
    public_distill = subparsers.add_parser(
        "prepare-public-distillation",
        help="convert the public semantic-task pool into teacher-ready CID tasks",
    )
    public_distill.add_argument("--data", default="data/generated/public-task-pool-v1.jsonl")
    public_distill.add_argument(
        "--tasks-output", default="data/generated/public-teacher-tasks-v1.train.jsonl"
    )
    public_distill.add_argument(
        "--requests-output", default="data/generated/public-teacher-requests-v1.train.jsonl"
    )
    public_distill.add_argument(
        "--manifest-output", default="data/generated/public-teacher-v1.train.manifest.json"
    )
    public_distill.add_argument(
        "--causal-jobs-output", default="data/generated/public-teacher-causal-v1.train.jsonl"
    )
    public_distill.add_argument("--split", choices=("train", "validation", "test"), default="train")
    public_distill.add_argument("--seed", type=int, default=20260809)
    public_distill.add_argument("--unnecessary-tool-fraction", type=float, default=0.10)
    wave_export = subparsers.add_parser(
        "teacher-wave-export",
        help="export the next causally visible teacher stage for each incomplete task",
    )
    wave_export.add_argument("--jobs", required=True)
    wave_export.add_argument("--state", required=True)
    wave_export.add_argument("--output", required=True)
    wave_export.add_argument("--max-requests", type=int)
    wave_import = subparsers.add_parser(
        "teacher-wave-import",
        help="validate and persist teacher responses for one exported causal wave",
    )
    wave_import.add_argument("--jobs", required=True)
    wave_import.add_argument("--requests", required=True)
    wave_import.add_argument("--responses", required=True)
    wave_import.add_argument("--state", required=True)
    wave_import.add_argument("--rejects-output")
    wave_finalize = subparsers.add_parser(
        "teacher-wave-finalize",
        help="assemble complete causal teacher state into TeacherPlan JSONL",
    )
    wave_finalize.add_argument("--tasks", required=True)
    wave_finalize.add_argument("--jobs", required=True)
    wave_finalize.add_argument("--state", required=True)
    wave_finalize.add_argument("--output", required=True)
    wave_status = subparsers.add_parser(
        "teacher-wave-status",
        help="summarize causal teacher production progress and next phases",
    )
    wave_status.add_argument("--jobs", required=True)
    wave_status.add_argument("--state", required=True)
    agent_checkout = subparsers.add_parser(
        "teacher-agent-checkout",
        help="stage a compact resumable causal-teacher batch for an interactive LSM agent",
    )
    agent_checkout.add_argument("--jobs", required=True)
    agent_checkout.add_argument("--state", required=True)
    agent_checkout.add_argument("--workspace", default=".cid/teacher-agent")
    agent_checkout.add_argument("--max-requests", type=int, default=8)
    agent_commit = subparsers.add_parser(
        "teacher-agent-commit",
        help="validate and persist responses from the current interactive teacher batch",
    )
    agent_commit.add_argument("--workspace", default=".cid/teacher-agent")
    benchmark = subparsers.add_parser(
        "benchmark",
        help="run a neural CID checkpoint on deterministic replay trajectories",
    )
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument(
        "--checkpoint-kind",
        choices=("stage-a", "stage-b"),
        default="stage-a",
    )
    benchmark.add_argument("--model", default="GSAI-ML/iLLaDA-8B-Base")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--summary-output", required=True)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    benchmark.add_argument("--denoising-steps", type=int, default=8)
    benchmark.add_argument("--max-steps", type=int, default=32)
    benchmark.add_argument("--max-examples", type=int)
    benchmark.add_argument("--progress-every", type=int, default=10)
    benchmark.add_argument("--seed-teacher-state", action="store_true")
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
    train.add_argument("--rollout-horizon", type=int, default=3)
    train.add_argument("--teacher-forcing-epochs", type=int, default=1)
    train.add_argument("--rollout-ramp-epochs", type=int, default=2)
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
    train_full = subparsers.add_parser(
        "train-full",
        help="run Stage B full-parameter iLLaDA training with FSDP FULL_SHARD",
    )
    train_full.add_argument("--data", required=True)
    train_full.add_argument("--output-dir", required=True)
    train_full.add_argument("--model", default="GSAI-ML/iLLaDA-8B-Base")
    train_full.add_argument("--resume")
    train_full.add_argument("--init-cid-checkpoint")
    train_full.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    train_full.add_argument("--epochs", type=int, default=1)
    train_full.add_argument("--learning-rate", type=float, default=1e-5)
    train_full.add_argument("--weight-decay", type=float, default=0.01)
    train_full.add_argument("--micro-batch-size", type=int, default=1)
    train_full.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_full.add_argument("--max-grad-norm", type=float, default=1.0)
    train_full.add_argument("--timestep-min", type=float, default=0.05)
    train_full.add_argument("--timestep-max", type=float, default=1.0)
    train_full.add_argument("--rollout-horizon", type=int, default=3)
    train_full.add_argument("--teacher-forcing-epochs", type=int, default=1)
    train_full.add_argument("--rollout-ramp-epochs", type=int, default=2)
    train_full.add_argument("--thought-capacity", type=int, default=8)
    train_full.add_argument("--max-display-tokens", type=int, default=1024)
    train_full.add_argument("--seed", type=int, default=0)
    train_full.add_argument("--max-examples", type=int)
    train_full.add_argument("--no-shuffle", action="store_true")
    train_full.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.command == "demo":
        asyncio.run(_run_demo())
    elif args.command == "generate-synthetic":
        _generate_synthetic(args)
    elif args.command == "build-computational-training":
        _build_computational_training(args)
    elif args.command == "build-correction-training":
        _build_correction_training(args)
    elif args.command == "build-symbolic-training":
        _build_symbolic_training(args)
    elif args.command == "build-self-identity-training":
        _build_self_identity_training(args)
    elif args.command == "build-multilingual-training":
        _build_multilingual_training(args)
    elif args.command == "prepare-distillation":
        _prepare_distillation(args)
    elif args.command == "compile-distillation":
        _compile_distillation(args)
    elif args.command == "review-distillation":
        _review_distillation(args)
    elif args.command == "dataset-manifest":
        _dataset_manifest(args)
    elif args.command == "materialize-trajectory-mixture":
        _materialize_trajectory_mixture(args)
    elif args.command == "build-public-task-pool":
        _build_public_task_pool(args)
    elif args.command == "prepare-public-distillation":
        _prepare_public_distillation(args)
    elif args.command == "teacher-wave-export":
        _teacher_wave_export(args)
    elif args.command == "teacher-wave-import":
        _teacher_wave_import(args)
    elif args.command == "teacher-wave-finalize":
        _teacher_wave_finalize(args)
    elif args.command == "teacher-wave-status":
        _teacher_wave_status(args)
    elif args.command == "teacher-agent-checkout":
        _teacher_agent_checkout(args)
    elif args.command == "teacher-agent-commit":
        _teacher_agent_commit(args)
    elif args.command == "benchmark":
        _benchmark(args)
    elif args.command == "train":
        _train_stage_a(args)
    elif args.command == "train-full":
        _train_stage_b(args)


if __name__ == "__main__":
    main()
