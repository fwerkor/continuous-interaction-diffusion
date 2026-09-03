from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from cid.accelerator import (
    configure_torch_accelerator,
    distributed_backend,
    resolve_torch_device_type,
    torch_device_available,
    wrap_npu_autocast,
    wrap_torch_autocast,
)
from cid.composed_training import ComposedTrainingConfig, build_composed_distillation
from cid.computational_training import (
    ComputationalTrainingConfig,
    build_computational_training,
)
from cid.contracts import FreshnessDemand, InformationNeed, ModelContext, ModelUpdate
from cid.correction_training import CorrectionTrainingConfig, build_correction_training
from cid.data import (
    TrajectoryExample,
    TrajectoryExampleIndex,
    dump_jsonl,
    index_training_and_validation_jsonl,
    index_training_jsonl,
    load_jsonl,
)
from cid.dataset import dump_dataset_manifest, inspect_dataset, validate_neural_training_contract
from cid.defaults import (
    DEFAULT_ALLOCATION_THRESHOLD,
    DEFAULT_ANCHOR_PRESENCE_THRESHOLD,
    DEFAULT_ARGUMENT_PRESENCE_THRESHOLD,
    DEFAULT_BINDING_THRESHOLD,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_DISPLAY_REVISION_FRACTION,
    DEFAULT_DISPLAY_REVISION_MARGIN,
    DEFAULT_LINK_PRESENCE_THRESHOLD,
    DEFAULT_MATERIALIZED_MAX_AGE_S,
    DEFAULT_MAX_ALLOCATIONS_PER_STEP,
    DEFAULT_NEED_TARGET_CELL_THRESHOLD,
    DEFAULT_NEED_TARGET_DISPLAY_THRESHOLD,
    DEFAULT_NEED_THRESHOLD,
    DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
)
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
from cid.tool_restraint_training import (
    ToolRestraintTrainingConfig,
    build_tool_restraint_distillation,
)


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


def _build_composed_training(args: argparse.Namespace) -> None:
    manifest = build_composed_distillation(
        output_dir=args.output_dir,
        reference_manifest_output=args.reference_manifest_output,
        config=ComposedTrainingConfig(
            count_per_family=args.count_per_family,
            seed=args.seed,
            variants_per_task=args.variants_per_task,
            thought_capacity=args.thought_capacity,
            min_delay_steps=args.min_delay_steps,
            max_delay_steps=args.max_delay_steps,
        ),
    )
    print(
        f"tasks={manifest['semantic_tasks']} trajectories={manifest['compiled_trajectories']} "
        f"transitions={manifest['compiled_transitions']}"
    )


def _build_tool_restraint_training(args: argparse.Namespace) -> None:
    manifest = build_tool_restraint_distillation(
        source_tasks_path=args.source_tasks,
        source_plans_path=args.source_plans,
        output_dir=args.output_dir,
        reference_manifest_output=args.reference_manifest_output,
        config=ToolRestraintTrainingConfig(
            count=args.count,
            seed=args.seed,
            thought_capacity=args.thought_capacity,
        ),
    )
    print(
        f"tasks={manifest['semantic_tasks']} trajectories={manifest['compiled_trajectories']} "
        f"mode=tools_available_unnecessary"
    )


def _build_deep_tool_restraint_training(args: argparse.Namespace) -> None:
    from cid.deep_restraint_training import (
        DeepToolRestraintConfig,
        build_deep_tool_restraint_distillation,
    )

    manifest = build_deep_tool_restraint_distillation(
        source_tasks_path=args.source_tasks,
        source_plans_path=args.source_plans,
        output_dir=args.output_dir,
        reference_manifest_output=args.reference_manifest_output,
        config=DeepToolRestraintConfig(
            count_per_bucket=args.count_per_bucket,
            min_dependency_depth=args.min_dependency_depth,
            capacity_buckets=tuple(args.capacity_buckets),
            seed=args.seed,
        ),
    )
    print(
        f"tasks={manifest['semantic_tasks']} trajectories={manifest['compiled_trajectories']} "
        f"transitions={manifest['compiled_transitions']} "
        f"mode=tools_available_unnecessary"
    )


def _build_natural_interaction_training(args: argparse.Namespace) -> None:
    from cid.natural_interaction_training import (
        NaturalInteractionConfig,
        build_natural_interaction_augmentation,
    )

    manifest = build_natural_interaction_augmentation(
        source_pairs=(
            (args.public_tasks, args.public_plans),
            (args.interaction_tasks, args.interaction_plans),
        ),
        output_dir=args.output_dir,
        reference_manifest_output=args.reference_manifest_output,
        config=NaturalInteractionConfig(
            thought_capacity=args.thought_capacity,
            variants_per_task=args.variants_per_task,
            min_delay_steps=args.min_delay_steps,
            max_delay_steps=args.max_delay_steps,
            seed=args.seed,
        ),
    )
    print(
        f"tasks={manifest['semantic_tasks']} trajectories={manifest['compiled_trajectories']} "
        f"long_form_targets={manifest['long_form_targets']}"
    )


def _build_natural_public_training(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from cid.natural_public_training import (
        NaturalPublicBuildConfig,
        build_natural_public_component,
        collect_unique_prompts_from_trajectories,
    )

    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    source_spec = next(item for item in registry["sources"] if item["id"] == args.source)
    quota = int(source_spec["quota"])
    output_dir = args.output_dir or f"data/generated/natural-public-v1/{args.source}"
    reference_manifest_output = (
        args.reference_manifest_output
        or f"data/natural-public-{args.source}-v1.reference-manifest.json"
    )

    exclude_prompts: list[str] = []
    for path in args.exclude_trajectories:
        exclude_prompts.extend(collect_unique_prompts_from_trajectories(path))
    variants = 1 if args.source == "oasst1" else args.variants_per_task
    manifest = build_natural_public_component(
        registry_path=args.registry,
        output_dir=output_dir,
        reference_manifest_output=reference_manifest_output,
        config=NaturalPublicBuildConfig(
            source=args.source,
            quota=quota,
            seed=args.seed,
            thought_capacity=args.thought_capacity,
            variants_per_task=variants,
            min_delay_steps=args.min_delay_steps,
            max_delay_steps=args.max_delay_steps,
        ),
        exclude_prompts=exclude_prompts,
        cache_dir=args.cache_dir,
    )
    print(
        f"source={args.source} tasks={manifest['semantic_tasks']} "
        f"trajectories={manifest['compiled_trajectories']} "
        f"transitions={manifest['compiled_training_transitions']}"
    )


def _build_surface_diverse_training(args: argparse.Namespace) -> None:
    from cid.surface_diversity_training import (
        SurfaceDiversityConfig,
        build_surface_diversified_distillation,
    )

    presets = {
        "composed": (
            "data/generated/composed-teacher-tasks-v1.jsonl",
            "data/generated/composed-teacher-plans-v1.accepted.jsonl",
            "data/generated/composed-v2",
            "data/composed-teacher-v2.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="composed-tool-reasoning-v2",
                file_stem="composed-v2",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
            ),
        ),
        "long-horizon": (
            "data/generated/long-horizon-v1/long-horizon-teacher-tasks-v1.jsonl",
            "data/generated/long-horizon-v1/long-horizon-teacher-plans-v1.accepted.jsonl",
            "data/generated/long-horizon-v2",
            "data/long-horizon-teacher-v2.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="long-horizon-tool-reasoning-v2",
                file_stem="long-horizon-v2",
                thought_capacity=12,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=5,
                seed=args.seed,
                max_tasks=args.max_tasks,
            ),
        ),
        "mechanism-v3": (
            "data/generated/mechanism-teacher-tasks-v1.jsonl",
            "data/generated/mechanism-teacher-plans-v1.accepted.jsonl",
            "data/generated/mechanism-v3",
            "data/mechanism-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="mechanism-teacher-v3",
                file_stem="mechanism-v3",
                thought_capacity=8,
                variants_per_task=4,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_semantic_text=True,
            ),
        ),
        "computational-v3": (
            "data/generated/computational-teacher-tasks-v1.jsonl",
            "data/generated/computational-teacher-plans-v1.accepted.jsonl",
            "data/generated/computational-v3",
            "data/computational-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="computational-teacher-v3",
                file_stem="computational-v3",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_semantic_text=True,
            ),
        ),
        "symbolic-v3": (
            "data/generated/symbolic-teacher-tasks-v1.jsonl",
            "data/generated/symbolic-teacher-plans-v1.accepted.jsonl",
            "data/generated/symbolic-v3",
            "data/symbolic-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="symbolic-teacher-v3",
                file_stem="symbolic-v3",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_semantic_text=True,
            ),
        ),
        "correction-v3": (
            "data/generated/correction-teacher-tasks-v1.jsonl",
            "data/generated/correction-teacher-plans-v1.accepted.jsonl",
            "data/generated/correction-v3",
            "data/correction-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="speculative-local-correction-v3",
                file_stem="correction-v3",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_semantic_text=True,
            ),
        ),
        "composed-v3": (
            "data/generated/composed-v2/composed-v2-teacher-tasks.jsonl",
            "data/generated/composed-v2/composed-v2-teacher-plans.accepted.jsonl",
            "data/generated/composed-v3",
            "data/composed-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="composed-tool-reasoning-v3",
                file_stem="composed-v3",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_prompt=False,
                diversify_semantic_text=True,
            ),
        ),
        "long-horizon-v3": (
            "data/generated/long-horizon-v2/long-horizon-v2-teacher-tasks.jsonl",
            "data/generated/long-horizon-v2/long-horizon-v2-teacher-plans.accepted.jsonl",
            "data/generated/long-horizon-v3",
            "data/long-horizon-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="long-horizon-tool-reasoning-v3",
                file_stem="long-horizon-v3",
                thought_capacity=12,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=5,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_prompt=False,
                diversify_semantic_text=True,
            ),
        ),
        "logic-v3": (
            "data/generated/logic-teacher-tasks-v1.jsonl",
            "data/generated/logic-teacher-plans-v1.accepted.jsonl",
            "data/generated/logic-v3",
            "data/logic-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="complex-logic-reasoning-v3",
                file_stem="logic-v3",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_prompt=False,
                diversify_semantic_text=True,
            ),
        ),
        "self-identity-v3": (
            "data/generated/self-identity-teacher-tasks-v1.jsonl",
            "data/generated/self-identity-teacher-plans-v1.accepted.jsonl",
            "data/generated/self-identity-v3",
            "data/self-identity-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="cid-self-identity-v3",
                file_stem="self-identity-v3",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_prompt=False,
                diversify_semantic_text=True,
            ),
        ),
        "multilingual-v3": (
            "data/generated/multilingual-v1/teacher-tasks-v1.jsonl",
            "data/generated/multilingual-v1/teacher-plans-v1.accepted.jsonl",
            "data/generated/multilingual-v3",
            "data/multilingual-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="multilingual-v3",
                file_stem="multilingual-v3",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_semantic_text=True,
            ),
        ),
        "compositional-v3": (
            "data/generated/compositional-teacher-tasks-v1.jsonl",
            "data/generated/compositional-teacher-plans-v1.accepted.jsonl",
            "data/generated/compositional-v3",
            "data/compositional-teacher-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="compositional-longtail-reasoning-v3",
                file_stem="compositional-v3",
                thought_capacity=128,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_prompt=False,
                diversify_semantic_text=True,
            ),
        ),
        "deep-restraint-v3": (
            "data/generated/deep-tool-restraint-v1/deep-tool-restraint-teacher-tasks-v1.jsonl",
            "data/generated/deep-tool-restraint-v1/deep-tool-restraint-teacher-plans-v1.accepted.jsonl",
            "data/generated/deep-restraint-v3",
            "data/deep-tool-restraint-v3.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="deep-tool-restraint-v3",
                file_stem="deep-restraint-v3",
                thought_capacity=128,
                variants_per_task=1,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=3,
                diversify_prompt=False,
                diversify_semantic_text=True,
            ),
        ),
        "logic-v4": (
            "data/generated/logic-teacher-tasks-v1.jsonl",
            "data/generated/logic-teacher-plans-v1.accepted.jsonl",
            "data/generated/logic-v4",
            "data/logic-teacher-v4.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="complex-logic-reasoning-v4",
                file_stem="logic-v4",
                thought_capacity=8,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=4,
                diversify_prompt=False,
                diversify_semantic_text=True,
                rewrite_semantic_text=True,
            ),
        ),
        "compositional-v4": (
            "data/generated/compositional-teacher-tasks-v1.jsonl",
            "data/generated/compositional-teacher-plans-v1.accepted.jsonl",
            "data/generated/compositional-v4",
            "data/compositional-teacher-v4.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="compositional-longtail-reasoning-v4",
                file_stem="compositional-v4",
                thought_capacity=128,
                variants_per_task=2,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=4,
                diversify_prompt=False,
                diversify_semantic_text=True,
                rewrite_semantic_text=True,
            ),
        ),
        "deep-restraint-v4": (
            "data/generated/deep-tool-restraint-v1/deep-tool-restraint-teacher-tasks-v1.jsonl",
            "data/generated/deep-tool-restraint-v1/deep-tool-restraint-teacher-plans-v1.accepted.jsonl",
            "data/generated/deep-restraint-v4",
            "data/deep-tool-restraint-v4.reference-manifest.json",
            SurfaceDiversityConfig(
                component_name="deep-tool-restraint-v4",
                file_stem="deep-restraint-v4",
                thought_capacity=128,
                variants_per_task=1,
                min_delay_steps=1,
                max_delay_steps=4,
                seed=args.seed,
                max_tasks=args.max_tasks,
                surface_version=4,
                diversify_prompt=False,
                diversify_semantic_text=True,
                rewrite_semantic_text=True,
            ),
        ),
    }
    source_tasks, source_plans, output_dir, reference, config = presets[args.component]
    manifest = build_surface_diversified_distillation(
        source_tasks_path=source_tasks,
        source_plans_path=source_plans,
        output_dir=output_dir,
        reference_manifest_output=reference,
        config=config,
    )
    print(
        f"tasks={manifest['semantic_tasks']} trajectories={manifest['compiled_trajectories']} "
        f"signatures={manifest['normalized_prompt_signatures']} output={output_dir}"
    )


def _build_long_horizon_training(args: argparse.Namespace) -> None:
    from cid.long_horizon_training import (
        LongHorizonTrainingConfig,
        build_long_horizon_distillation,
    )

    manifest = build_long_horizon_distillation(
        output_dir=args.output_dir,
        reference_manifest_output=args.reference_manifest_output,
        config=LongHorizonTrainingConfig(
            count_per_family=args.count_per_family,
            seed=args.seed,
            variants_per_task=args.variants_per_task,
            thought_capacity=args.thought_capacity,
            min_delay_steps=args.min_delay_steps,
            max_delay_steps=args.max_delay_steps,
        ),
    )
    print(
        f"tasks={manifest['semantic_tasks']} trajectories={manifest['compiled_trajectories']} "
        f"transitions={manifest['compiled_transitions']} "
        f"depth4_plus={manifest['depth_4_plus_tasks']}"
    )


def _build_compositional_training(args: argparse.Namespace) -> None:
    from cid.compositional_training import (
        CompositionalTrainingConfig,
        build_compositional_training_streaming,
    )

    result = build_compositional_training_streaming(
        args.output_dir,
        CompositionalTrainingConfig(
            seed=args.seed,
            variants_per_task=args.variants_per_task,
            probe_variants_per_task=args.probe_variants_per_task,
        ),
    )
    print(
        f"train_tasks={result['train_tasks']} "
        f"train_trajectories={result['train_trajectories']} "
        f"probe_tasks={result['probe_tasks']} "
        f"max_live_cells={result['train_manifest']['audit']['max_live_cells']} "
        f"output_dir={args.output_dir}"
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


def _migrate_dataset_contract_v3(args: argparse.Namespace) -> None:
    from cid.dataset_contract_v3 import migrate_dataset_contract_v3

    manifest = migrate_dataset_contract_v3(args.data, args.output, args.manifest_output)
    print(
        f"examples={manifest['examples']} bindings={manifest['bindings']} "
        f"multi_cell={manifest['multi_cell_bindings']} "
        f"display_routed={manifest['display_routed_bindings']} "
        f"sha256={manifest['sha256']} path={args.output}"
    )


def _build_contract_v3_validation(args: argparse.Namespace) -> None:
    from cid.validation_dataset import build_contract_v3_validation

    manifest = build_contract_v3_validation(
        args.reasoning_source,
        args.output,
        args.manifest_output,
        total_examples=args.examples,
        tool_examples=args.tool_examples,
        seed=args.seed,
    )
    print(
        f"examples={manifest['examples']} tool_examples={manifest['tool_examples']} "
        f"tool_fraction={manifest['tool_fraction']:.4f} sha256={manifest['sha256']} "
        f"path={args.output}"
    )


def _migrate_dataset_contract_v4(args: argparse.Namespace) -> None:
    from cid.dataset_contract_v4 import migrate_dataset_contract_v4

    manifest = migrate_dataset_contract_v4(
        args.data,
        args.output,
        args.manifest_output,
        curated_path=args.curated_data,
    )
    print(
        f"examples={manifest['examples']} base={manifest['base_examples']} "
        f"curated={manifest['curated_examples']} rewrites={manifest['status_rewrites']} "
        f"settle_steps={manifest['appended_settle_steps']} sha256={manifest['sha256']} "
        f"path={args.output}"
    )


def _build_contract_v4_validation(args: argparse.Namespace) -> None:
    from cid.validation_dataset_v4 import build_contract_v4_validation

    manifest = build_contract_v4_validation(
        args.reasoning_source,
        args.output,
        args.manifest_output,
        total_examples=args.examples,
        tool_examples=args.tool_examples,
        curated_examples=args.curated_examples,
        seed=args.seed,
    )
    print(
        f"examples={manifest['examples']} tool_examples={manifest['tool_examples']} "
        f"tool_fraction={manifest['tool_fraction']:.4f} sha256={manifest['sha256']} "
        f"path={args.output}"
    )


def _build_curated_v4_training(args: argparse.Namespace) -> None:
    from cid.curated_v4_training import CuratedV4Config, build_curated_v4_training

    manifest = build_curated_v4_training(
        args.output,
        args.manifest_output,
        CuratedV4Config(
            count_per_family=args.count_per_family,
            seed=args.seed,
            thought_capacity=args.thought_capacity,
            training_weight=args.training_weight,
        ),
    )
    print(
        f"examples={manifest['examples']} archetypes={manifest['hand_authored_archetypes']} "
        f"training_weight={manifest['training_weight']} sha256={manifest['sha256']} "
        f"path={args.output}"
    )


def _materialize_trajectory_mixture(args: argparse.Namespace) -> None:
    from cid.trajectory_mixture import materialize_trajectory_mixture

    manifest = materialize_trajectory_mixture(args.spec, args.output, args.manifest_output)
    print(
        f"examples={manifest['examples']} transitions={manifest['transitions']} "
        f"sha256={manifest['sha256']} path={args.output} manifest={args.manifest_output}"
    )


def _audit_training_data(args: argparse.Namespace) -> None:
    from cid.training_audit import audit_training_trajectories

    audit = audit_training_trajectories(args.data)
    print(json.dumps(audit.to_dict(), ensure_ascii=False, sort_keys=True))
    if args.strict and not audit.ok:
        raise SystemExit(2)


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
        CIDMaterializerConfig,
        ILLaDACIDAdapter,
        ILLaDACIDConfig,
        load_cid_adapter_checkpoint,
        load_cid_adapter_from_pretrained,
        load_stage_b_model_checkpoint,
        load_stage_b_semantic_encoder,
        wrap_stage_b_fsdp,
    )
    from cid.model.benchmark import run_neural_benchmark_case
    from cid.model.encoding import ILLaDATextEncoder
    from cid.model.loading import pretrained_revision

    checkpoint = Path(args.checkpoint)
    stage_b = args.checkpoint_kind == "stage-b"
    stage_b_sharded = stage_b and checkpoint.is_dir()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = stage_b_sharded
    device_type = resolve_torch_device_type(torch, args.device)
    if stage_b_sharded:
        if world_size < 2:
            raise RuntimeError(
                "sharded Stage B benchmark must run under multi-accelerator torchrun"
            )
        if device_type == "cpu":
            raise RuntimeError("sharded Stage B benchmark requires CUDA GPUs or Ascend NPUs")
        configure_torch_accelerator(torch, device_type, local_rank)
        dist.init_process_group(backend=distributed_backend(device_type))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(device_type, local_rank)
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        adapter_config = ILLaDACIDConfig(**metadata["adapter_config"])
        semantic_pooling = str(metadata.get("semantic_pooling", "mean-v1"))
    else:
        if world_size > 1:
            raise RuntimeError("unsharded benchmark checkpoints are single-process; omit torchrun")
        configure_torch_accelerator(torch, device_type, local_rank)
        device = torch.device(device_type)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adapter_config = ILLaDACIDConfig(**raw["adapter_config"])
        semantic_pooling = str(
            raw.get("trainer_config", {}).get("semantic_pooling", "mean-v1")
        )

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
    revision = pretrained_revision(args.model)
    if revision is not None:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)

    try:

        def load_adapter() -> ILLaDACIDAdapter:
            return load_cid_adapter_from_pretrained(
                args.model,
                config=adapter_config,
                freeze_backbone=True,
                torch_dtype=torch.float32 if stage_b else dtype,
                low_cpu_mem_usage=True,
            ).to(device)

        adapter = None
        if stage_b_sharded:
            for loading_rank in range(world_size):
                if rank == loading_rank:
                    adapter = load_adapter()
                dist.barrier()
        else:
            adapter = load_adapter()
        if adapter is None:
            raise RuntimeError("failed to load benchmark CID adapter")

        if stage_b:
            has_saved_semantic_snapshot = (
                isinstance(metadata.get("semantic_embedding_snapshot"), dict)
                if stage_b_sharded
                else "semantic_embedding_snapshot" in raw
            )
            legacy_text_encoder = (
                None
                if has_saved_semantic_snapshot
                else ILLaDATextEncoder.from_frozen_snapshot(
                    adapter,
                    tokenizer,
                    device=device,
                    dtype=dtype,
                    pooling_mode=semantic_pooling,
                )
            )
            adapter.set_backbone_trainable(True)
            if device_type == "npu":
                adapter.set_device_value_validation(False)
            if stage_b_sharded:
                forward_model = wrap_stage_b_fsdp(
                    adapter,
                    device_id=device,
                    compute_dtype=dtype,
                )
                saved_text_encoder = load_stage_b_model_checkpoint(
                    forward_model,
                    adapter,
                    checkpoint,
                    tokenizer=tokenizer,
                    semantic_device=device,
                    semantic_embedding_device="cpu",
                )
                text_encoder = saved_text_encoder or legacy_text_encoder
            else:
                text_encoder = (
                    load_stage_b_semantic_encoder(
                        adapter,
                        tokenizer,
                        checkpoint,
                        device=device,
                        embedding_device="cpu",
                    )
                    if "semantic_embedding_snapshot" in raw
                    else legacy_text_encoder
                )
            if text_encoder is None:
                raise RuntimeError("failed to restore Stage B semantic encoder")
            if not stage_b_sharded:
                load_cid_adapter_checkpoint(adapter, checkpoint)
                forward_model = (
                    wrap_npu_autocast(torch, adapter, dtype=dtype)
                    if device_type == "npu"
                    else adapter
                )
        else:
            load_cid_adapter_checkpoint(adapter, checkpoint)
            text_encoder = ILLaDATextEncoder(
                adapter, tokenizer, pooling_mode=semantic_pooling
            )
            forward_model = adapter

        adapter.eval()
        forward_model.eval()
        examples = load_jsonl(args.data)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        if not examples:
            raise ValueError("benchmark dataset is empty")

        materializer_config = CIDMaterializerConfig(
            allocation_threshold=args.allocation_threshold,
            convergence_threshold=args.convergence_threshold,
            need_threshold=args.need_threshold,
            need_target_cell_threshold=args.need_target_cell_threshold,
            need_target_display_threshold=args.need_target_display_threshold,
            argument_presence_threshold=args.argument_presence_threshold,
            anchor_presence_threshold=args.anchor_presence_threshold,
            link_presence_threshold=args.link_presence_threshold,
            retrieval_similarity_threshold=args.retrieval_similarity_threshold,
            max_allocations_per_step=args.max_allocations_per_step,
            max_age_s=args.max_age_s,
        )
        runtime_config = RuntimeConfig(
            max_steps=args.max_steps,
            max_wall_time_s=args.max_wall_time_s,
            binding_threshold=args.binding_threshold,
            idle_yield_s=args.idle_yield_s,
            reclamation_grace_steps=args.reclamation_grace_steps,
            reclamation_low_watermark=args.reclamation_low_watermark,
            reclamation_target_watermark=args.reclamation_target_watermark,
        )

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
                    display_revision_fraction=args.display_revision_fraction,
                    display_revision_margin=args.display_revision_margin,
                    materializer_config=materializer_config,
                    runtime_config=runtime_config,
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
                f"coverage={summary.observation_coverage:.4f} "
                f"tool_wait={summary.tool_wait_ratio:.4f} "
                f"hidden={summary.latency_hidden_ratio:.4f} output={output}"
            )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _trajectory_split(example: TrajectoryExample) -> str | None:
    split = str(example.metadata.get("split", "")).strip().lower()
    return split if split in {"train", "validation", "test"} else None


def _load_explicit_validation_examples(
    validation_data_path: str | Path,
    *,
    max_validation_examples: int | None,
) -> tuple[TrajectoryExample, ...]:
    validation_source = load_jsonl(validation_data_path)
    validation_labels = tuple(_trajectory_split(example) for example in validation_source)
    if any(label is not None for label in validation_labels):
        validation_examples = tuple(
            example
            for example, label in zip(validation_source, validation_labels, strict=True)
            if label == "validation"
        )
        if not validation_examples:
            raise ValueError("--validation-data contains split labels but no validation examples")
    else:
        validation_examples = validation_source
    if max_validation_examples is not None:
        validation_examples = validation_examples[:max_validation_examples]
    return validation_examples


def _index_train_and_load_validation_examples(
    data_path: str | Path,
    *,
    validation_data_path: str | Path | None,
    max_examples: int | None,
    max_validation_examples: int | None,
) -> tuple[tuple[TrajectoryExampleIndex, ...], tuple[TrajectoryExample, ...]]:
    """Keep training data compact while materializing only held-out trajectories."""

    if max_validation_examples is not None and max_validation_examples < 0:
        raise ValueError("max_validation_examples must be non-negative when set")
    if validation_data_path is None:
        training_examples, validation_examples = index_training_and_validation_jsonl(
            data_path,
            max_examples=max_examples,
            max_validation_examples=max_validation_examples,
        )
    else:
        training_examples = index_training_jsonl(data_path, max_examples=max_examples)
        validation_examples = _load_explicit_validation_examples(
            validation_data_path,
            max_validation_examples=max_validation_examples,
        )
    validation_ids = {example.example_id for example in validation_examples}
    overlapping_ids = validation_ids.intersection(
        example.example_id for example in training_examples
    )
    if overlapping_ids:
        sample = sorted(overlapping_ids)[:3]
        raise ValueError(
            "training and validation data overlap by example_id: " + ", ".join(sample)
        )
    return training_examples, validation_examples


def _load_train_and_validation_examples(
    data_path: str | Path,
    *,
    validation_data_path: str | Path | None,
    max_examples: int | None,
    max_validation_examples: int | None,
) -> tuple[tuple[TrajectoryExample, ...], tuple[TrajectoryExample, ...]]:
    examples = load_jsonl(data_path)
    has_split_labels = any(_trajectory_split(example) is not None for example in examples)
    if has_split_labels:
        training_examples = tuple(
            example
            for example in examples
            if _trajectory_split(example) in {None, "train"}
        )
        inline_validation = tuple(
            example for example in examples if _trajectory_split(example) == "validation"
        )
    else:
        training_examples = examples
        inline_validation = ()

    if validation_data_path is not None:
        validation_examples = _load_explicit_validation_examples(
            validation_data_path,
            max_validation_examples=max_validation_examples,
        )
    else:
        validation_examples = inline_validation

    if max_examples is not None:
        training_examples = training_examples[:max_examples]
    if max_validation_examples is not None and validation_data_path is None:
        validation_examples = validation_examples[:max_validation_examples]
    if not training_examples:
        raise ValueError("training data contains no train examples")

    training_ids = {example.example_id for example in training_examples}
    overlapping_ids = training_ids.intersection(
        example.example_id for example in validation_examples
    )
    if overlapping_ids:
        sample = sorted(overlapping_ids)[:3]
        raise ValueError(
            "training and validation data overlap by example_id: " + ", ".join(sample)
        )
    return training_examples, validation_examples


def _aggregate_validation_report(
    report,
    *,
    torch_module,
    dist_module,
    device,
    device_type: str,
    distributed: bool,
) -> dict[str, object]:
    component_names = tuple(sorted(report.component_mean_losses))
    behavior_names = tuple(sorted(report.behavior_counts))
    values = [
        report.mean_loss * report.transitions,
        report.raw_mean_loss * report.transitions,
        float(report.transitions),
    ]
    values.extend(
        report.component_mean_losses[name] * report.transitions
        for name in component_names
    )
    values.extend(report.behavior_counts[name] for name in behavior_names)
    aggregate = torch_module.tensor(
        values,
        device=device,
        dtype=torch_module.float32 if device_type == "npu" else torch_module.float64,
    )
    if distributed:
        dist_module.all_reduce(aggregate)
    transition_count = int(aggregate[2])
    component_offset = 3
    behavior_offset = component_offset + len(component_names)
    component_losses = {
        name: float(aggregate[component_offset + index] / aggregate[2])
        for index, name in enumerate(component_names)
    }
    behavior_counts = {
        name: float(aggregate[behavior_offset + index])
        for index, name in enumerate(behavior_names)
    }

    def safe_ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator > 0.0 else None

    tp = behavior_counts.get("need_tp", 0.0)
    fp = behavior_counts.get("need_fp", 0.0)
    fn = behavior_counts.get("need_fn", 0.0)
    tn = behavior_counts.get("need_tn", 0.0)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0.0
        else None
    )
    behavior_metrics = {
        "need_precision": precision,
        "need_recall": recall,
        "need_f1": f1,
        "need_false_positive_rate": safe_ratio(fp, fp + tn),
        "source_accuracy": safe_ratio(
            behavior_counts.get("source_correct", 0.0),
            behavior_counts.get("source_total", 0.0),
        ),
        "convergence_accuracy": safe_ratio(
            behavior_counts.get("convergence_correct", 0.0),
            behavior_counts.get("convergence_total", 0.0),
        ),
        "lifecycle_accuracy": safe_ratio(
            behavior_counts.get("lifecycle_correct", 0.0),
            behavior_counts.get("lifecycle_total", 0.0),
        ),
        "display_token_accuracy": safe_ratio(
            behavior_counts.get("display_token_correct", 0.0),
            behavior_counts.get("display_token_total", 0.0),
        ),
        "materialized_display_token_accuracy": safe_ratio(
            behavior_counts.get("materialized_display_token_correct", 0.0),
            behavior_counts.get("materialized_display_token_total", 0.0),
        ),
        "materialized_display_exact_rate": safe_ratio(
            behavior_counts.get("materialized_display_exact", 0.0),
            behavior_counts.get("materialized_display_total", 0.0),
        ),
        "rollout_recovery_failure_rate": safe_ratio(
            behavior_counts.get("rollout_recovery_failures", 0.0),
            behavior_counts.get("rollout_transition_total", 0.0),
        ),
    }
    return {
        "mean_loss": float(aggregate[0] / aggregate[2]),
        "raw_mean_loss": float(aggregate[1] / aggregate[2]),
        "transitions": transition_count,
        "component_losses": component_losses,
        "behavior_counts": behavior_counts,
        "behavior_metrics": behavior_metrics,
    }


def _replace_checkpoint_alias(
    alias: Path,
    target: Path,
    *,
    target_is_directory: bool,
) -> None:
    if alias.is_symlink() or alias.is_file():
        alias.unlink(missing_ok=True)
    elif alias.exists():
        shutil.rmtree(alias)
    alias.symlink_to(target.name, target_is_directory=target_is_directory)


def _stage_a_needs_legacy_resume_repair(
    *,
    data_order_version: int,
    windows_seen_in_epoch: int,
) -> bool:
    """Whether a partial Stage A checkpoint used the pre-v2 under-filled padding order."""

    return windows_seen_in_epoch > 0 and data_order_version < 2


def _stage_a_completed_epoch_data_order_version(data_order_version: int) -> int:
    """Upgrade legacy ordering only after its in-flight epoch has finished."""

    return max(data_order_version, 4)


def _train_stage_a(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from cid.model import (
        CIDTrainer,
        CIDTrainerConfig,
        CIDTrainerState,
        CIDTrainProgress,
        ILLaDACIDAdapter,
        ILLaDACIDConfig,
        ILLaDATrajectoryTensorizer,
        balance_rollout_windows_by_semantic_task,
        load_cid_adapter_from_pretrained,
        materialize_indexed_rollout_windows,
        shard_rollout_windows,
        trajectory_rollout_windows,
        wrap_stage_a_ddp,
    )
    from cid.model.encoding import ILLaDATextEncoder
    from cid.model.loading import pretrained_revision

    if args.thought_capacity != 128:
        raise ValueError("CID v1 Stage A requires --thought-capacity 128")
    if args.dtype == "fp16":
        raise ValueError(
            "Stage A training does not support --dtype fp16 without loss scaling; use bf16"
        )
    dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    adapter_config = ILLaDACIDConfig(
        max_thought_slots=args.thought_capacity,
        max_display_tokens=args.max_display_tokens,
        display_canvas_tokens=args.display_canvas_tokens,
    )
    dataset_manifest = inspect_dataset(args.data)
    validate_neural_training_contract(
        dataset_manifest,
        max_argument_slots=adapter_config.max_argument_slots,
        max_need_slots=adapter_config.max_need_slots,
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_type = resolve_torch_device_type(torch, args.device)
    configure_torch_accelerator(torch, device_type, local_rank)
    if distributed:
        dist.init_process_group(backend=distributed_backend(device_type))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = (
            f"{device_type}:{local_rank}"
            if device_type in {"cuda", "npu"}
            else "cpu"
        )
    else:
        device = device_type

    try:
        def load_adapter() -> ILLaDACIDAdapter:
            model = load_cid_adapter_from_pretrained(
                args.model,
                config=adapter_config,
                freeze_backbone=True,
                torch_dtype=dtype,
            )
            # Stage A freezes the low-precision backbone, so keep only the trainable
            # CID modules in FP32. Autocast below preserves low-precision compute while
            # AdamW updates and moments remain FP32 instead of being quantized away.
            model.set_cid_modules_dtype(torch.float32)
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
            raise RuntimeError("failed to load CID adapter on this training rank")
        if device_type == "npu":
            adapter.set_device_value_validation(False)
        if args.gradient_checkpointing:
            adapter.set_gradient_checkpointing(True)
        grouped_moe_layers = adapter.pack_frozen_moe_experts()

        tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
        revision = pretrained_revision(args.model)
        if revision is not None:
            tokenizer_kwargs["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
        text_encoder = ILLaDATextEncoder(
            adapter, tokenizer, pooling_mode=args.semantic_pooling
        )
        tensorizer = ILLaDATrajectoryTensorizer(
            adapter, tokenizer, text_encoder=text_encoder
        )
        forward_model = (
            wrap_stage_a_ddp(
                adapter,
                device_ids=(
                    [local_rank]
                    if str(device).split(":", 1)[0] in {"cuda", "npu"}
                    else None
                ),
            )
            if distributed
            else adapter
        )
        if dtype is not torch.float32:
            forward_model = wrap_torch_autocast(
                torch,
                forward_model,
                device_type=device_type,
                dtype=dtype,
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
                warmup_steps=args.warmup_steps,
                lr_decay_steps=args.lr_decay_steps,
                min_learning_rate_ratio=args.min_learning_rate_ratio,
                timestep_min=args.timestep_min,
                timestep_max=args.timestep_max,
                rollout_horizon=args.rollout_horizon,
                rollout_allocation_threshold=args.rollout_allocation_threshold,
                rollout_max_allocations_per_step=args.rollout_max_allocations_per_step,
                teacher_forcing_epochs=args.teacher_forcing_epochs,
                rollout_ramp_epochs=args.rollout_ramp_epochs,
                semantic_pooling=args.semantic_pooling,
                seed=args.seed,
            ),
            forward_model=forward_model,
        )
        if args.resume:
            trainer.load_checkpoint(
                args.resume,
                expected_dataset_sha256=dataset_manifest.sha256,
            )
        if distributed:
            trainer.reseed(args.seed + rank + trainer.state.transitions_seen * 104729)

        examples, validation_examples = _index_train_and_load_validation_examples(
            args.data,
            validation_data_path=args.validation_data,
            max_examples=args.max_examples,
            max_validation_examples=args.max_validation_examples,
        )
        windows = balance_rollout_windows_by_semantic_task(
            trajectory_rollout_windows(examples, max_horizon=args.rollout_horizon)
        )
        if not windows:
            raise ValueError("training data contains no adjacent thought transitions")
        validation_windows = balance_rollout_windows_by_semantic_task(
            trajectory_rollout_windows(
                validation_examples,
                max_horizon=args.rollout_horizon,
            )
        )
        transition_count_total = sum(len(window.source_steps) for window in windows)
        validation_transition_count_total = sum(
            len(window.source_steps) for window in validation_windows
        )
        local_validation_windows = (
            shard_rollout_windows(
                validation_windows,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                epoch=0,
                shuffle=False,
                micro_batch_size=args.micro_batch_size,
                portable_bucket_order=True,
            )
            if validation_windows
            else ()
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = (
            output_dir / f"train_metrics.rank-{rank:04d}.jsonl"
            if distributed
            else output_dir / "train_metrics.jsonl"
        )
        rank_zero_metrics_path = output_dir / "train_metrics.jsonl"
        validation_metrics_path = output_dir / "validation_metrics.jsonl"
        if args.log_every_steps <= 0:
            raise ValueError("--log-every-steps must be positive")
        if args.checkpoint_every_steps <= 0:
            raise ValueError("--checkpoint-every-steps must be positive")
        if args.physical_micro_batch_size is not None and args.physical_micro_batch_size <= 0:
            raise ValueError("--physical-micro-batch-size must be positive")
        trainable = sum(
            parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
        )
        if rank == 0:
            effective_batch = args.micro_batch_size * args.gradient_accumulation_steps * world_size
            print(
                f"device={device} world_size={world_size} dtype={args.dtype} "
                f"examples={len(examples)} transitions={transition_count_total} "
                f"validation_examples={len(validation_examples)} "
                f"validation_transitions={validation_transition_count_total} "
                f"trainable_parameters={trainable} effective_batch={effective_batch} "
                f"physical_micro_batch={args.physical_micro_batch_size or args.micro_batch_size} "
                f"grouped_moe_layers={grouped_moe_layers}"
            )

        first_epoch = trainer.state.epochs_completed + 1
        run_started = time.monotonic()
        next_checkpoint_step = (
            trainer.state.optimizer_steps // args.checkpoint_every_steps + 1
        ) * args.checkpoint_every_steps
        for epoch in range(first_epoch, args.epochs + 1):
            legacy_partial_resume = bool(
                args.resume
                and epoch == first_epoch
                and _stage_a_needs_legacy_resume_repair(
                    data_order_version=trainer.data_order_version,
                    windows_seen_in_epoch=trainer.state.rollout_windows_seen_in_epoch,
                )
            )
            local_windows = shard_rollout_windows(
                windows,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                epoch=epoch,
                shuffle=not args.no_shuffle,
                micro_batch_size=args.micro_batch_size,
                legacy_resume_padding=legacy_partial_resume,
                length_aware=trainer.data_order_version >= 2,
                zero_gradient_padding=trainer.data_order_version >= 3,
                portable_bucket_order=trainer.data_order_version >= 4,
            )
            total_local_windows = len(local_windows)
            resumed_windows = trainer.state.rollout_windows_seen_in_epoch
            if legacy_partial_resume and distributed and metrics_path.exists():
                rank_cursor = None
                with metrics_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            int(record.get("epoch", -1)) == epoch
                            and int(record.get("optimizer_steps", -1))
                            == trainer.state.optimizer_steps
                        ):
                            rank_cursor = int(record["windows_seen_in_epoch"])
                if rank_cursor is not None and rank_cursor != resumed_windows:
                    resumed_windows = rank_cursor
                    trainer.state = CIDTrainerState(
                        transitions_seen=trainer.state.transitions_seen,
                        optimizer_steps=trainer.state.optimizer_steps,
                        epochs_completed=trainer.state.epochs_completed,
                        rollout_windows_seen_in_epoch=resumed_windows,
                    )
            if resumed_windows:
                if resumed_windows >= total_local_windows:
                    raise ValueError(
                        "checkpoint rollout position is outside the current epoch shard"
                    )
                local_windows = local_windows[resumed_windows:]
                if legacy_partial_resume and distributed:
                    # Legacy Stage-A padding could leave some ranks one micro-batch
                    # further ahead than others at a checkpoint.  The repaired shard
                    # lengths are equal, but the rank-local cursors intentionally
                    # preserve that historical offset.  Pad only the *remaining*
                    # continuation so every rank executes the same number and sizes of
                    # DDP backward calls; otherwise the shorter ranks enter the epoch
                    # barrier while the longer ranks are still reducing gradients.
                    remaining = torch.tensor(
                        [len(local_windows)], device=device, dtype=torch.int64
                    )
                    dist.all_reduce(remaining, op=dist.ReduceOp.MAX)
                    target_remaining = int(remaining.item())
                    missing = target_remaining - len(local_windows)
                    if missing:
                        repair_source = local_windows
                        if not repair_source:
                            raise RuntimeError(
                                "cannot repair an empty legacy Stage-A resume shard"
                            )
                        local_windows = local_windows + tuple(
                            repair_source[index % len(repair_source)]
                            for index in range(missing)
                        )
                        total_local_windows = resumed_windows + len(local_windows)
                if rank == 0:
                    print(
                        f"resume epoch={epoch} optimizer_steps={trainer.state.optimizer_steps} "
                        f"windows_seen={resumed_windows}/{total_local_windows}",
                        flush=True,
                    )
            local_windows = materialize_indexed_rollout_windows(args.data, local_windows)
            rollout_probability = trainer.rollout_probability()

            def report_progress(
                progress: CIDTrainProgress,
                *,
                current_epoch: int = epoch,
                current_rollout_probability: float = rollout_probability,
                current_total_windows: int = total_local_windows,
            ) -> None:
                nonlocal next_checkpoint_step
                interval_transitions = progress.transitions
                mean_loss = progress.mean_loss
                raw_mean_loss = progress.raw_mean_loss
                windows_seen = progress.rollout_windows_seen_in_epoch
                record = {
                    "timestamp": time.time(),
                    "elapsed_seconds": time.monotonic() - run_started,
                    "epoch": current_epoch,
                    "rank": rank,
                    "world_size": world_size,
                    "aggregation": "local_rank",
                    "optimizer_steps": progress.optimizer_steps,
                    "interval_transitions": interval_transitions,
                    "mean_loss": mean_loss,
                    "weighted_mean_loss": mean_loss,
                    "raw_mean_loss": raw_mean_loss,
                    "component_losses": progress.component_mean_losses,
                    "learning_rate": progress.learning_rate,
                    "rollout_probability": current_rollout_probability,
                    "windows_seen_in_epoch": windows_seen,
                    "windows_total_in_epoch": current_total_windows,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if rank == 0:
                    if distributed:
                        with rank_zero_metrics_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(
                        f"progress epoch={current_epoch} "
                        f"optimizer_steps={progress.optimizer_steps} "
                        f"raw_loss={raw_mean_loss:.6f} weighted_loss={mean_loss:.6f} "
                        f"lr={progress.learning_rate:.6g} "
                        f"windows={windows_seen}/{current_total_windows}",
                        flush=True,
                    )

                checkpoint_due = progress.optimizer_steps >= next_checkpoint_step
                if checkpoint_due and windows_seen < current_total_windows:
                    if rank == 0:
                        checkpoint = output_dir / "stage-a-latest.pt"
                        trainer.save_checkpoint(
                            checkpoint,
                            dataset_sha256=dataset_manifest.sha256,
                        )
                        print(
                            f"checkpoint optimizer_steps={progress.optimizer_steps} "
                            f"path={checkpoint}",
                            flush=True,
                        )
                    while next_checkpoint_step <= progress.optimizer_steps:
                        next_checkpoint_step += args.checkpoint_every_steps

            report = trainer.train_rollout_windows(
                local_windows,
                epochs=1,
                shuffle=False,
                preserve_order=True,
                physical_micro_batch_size=args.physical_micro_batch_size,
                progress_every_optimizer_steps=args.log_every_steps,
                progress_callback=report_progress,
            )

            transition_count = report.transitions
            mean_loss = report.mean_loss
            raw_mean_loss = report.raw_mean_loss

            checkpoint = output_dir / f"stage-a-epoch-{epoch:04d}.pt"
            trainer.data_order_version = _stage_a_completed_epoch_data_order_version(
                trainer.data_order_version
            )
            if rank == 0:
                trainer.save_checkpoint(
                    checkpoint,
                    dataset_sha256=dataset_manifest.sha256,
                )
                step_alias = output_dir / f"stage-a-step-{trainer.state.optimizer_steps:08d}.pt"
                _replace_checkpoint_alias(
                    step_alias, checkpoint, target_is_directory=False
                )
                _replace_checkpoint_alias(
                    output_dir / "stage-a-latest.pt",
                    checkpoint,
                    target_is_directory=False,
                )
            if distributed:
                dist.barrier()

            validation_mean_loss = None
            if local_validation_windows:
                teacher_forced_report = trainer.evaluate_rollout_windows(
                    local_validation_windows,
                    seed=args.seed + 1_000_003,
                    rollout_probability=0.0,
                )
                teacher_forced = _aggregate_validation_report(
                    teacher_forced_report,
                    torch_module=torch,
                    dist_module=dist,
                    device=device,
                    device_type=device_type,
                    distributed=distributed,
                )
                free_rollout_report = trainer.evaluate_rollout_windows(
                    local_validation_windows,
                    seed=args.seed + 1_000_003,
                    rollout_probability=1.0,
                )
                free_rollout = _aggregate_validation_report(
                    free_rollout_report,
                    torch_module=torch,
                    dist_module=dist,
                    device=device,
                    device_type=device_type,
                    distributed=distributed,
                )
                validation_mean_loss = float(teacher_forced["mean_loss"])
                validation_raw_mean_loss = float(teacher_forced["raw_mean_loss"])
                validation_transitions = int(teacher_forced["transitions"])
                if rank == 0:
                    validation_record = {
                        "timestamp": time.time(),
                        "elapsed_seconds": time.monotonic() - run_started,
                        "epoch": epoch,
                        "optimizer_steps": trainer.state.optimizer_steps,
                        "validation_examples": len(validation_examples),
                        "validation_transitions": validation_transitions,
                        "validation_mean_loss": validation_mean_loss,
                        "validation_raw_mean_loss": validation_raw_mean_loss,
                        "validation_seed": args.seed + 1_000_003,
                        "objective": "teacher_forced_and_free_rollout_fixed_noise",
                        "teacher_forced": teacher_forced,
                        "free_rollout": free_rollout,
                        "checkpoint": str(checkpoint),
                    }
                    with validation_metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(validation_record, sort_keys=True) + "\n")

            if rank == 0:
                validation_text = (
                    "disabled"
                    if validation_mean_loss is None
                    else f"{validation_mean_loss:.6f}"
                )
                print(
                    f"epoch={epoch} transitions={transition_count} "
                    f"optimizer_steps={trainer.state.optimizer_steps} raw_loss={raw_mean_loss:.6f} "
                    f"weighted_loss={mean_loss:.6f} validation_loss={validation_text} "
                    f"rollout_probability={rollout_probability:.3f} checkpoint={checkpoint}",
                    flush=True,
                )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _stage_b_execution_target(
    requested_device: str,
    *,
    cuda_available: bool,
    npu_available: bool,
    world_size: int,
    local_rank: int,
    dtype: str,
    cpu_offload: bool,
) -> tuple[str, int | None, str]:
    """Resolve Stage B compute device and distributed backend."""
    if requested_device not in {"auto", "cuda", "npu", "cpu"}:
        raise ValueError(f"unsupported Stage B device {requested_device!r}")
    if world_size <= 0:
        raise ValueError("Stage B world size must be positive")
    if dtype != "bf16":
        raise ValueError(
            "Stage B training supports --dtype bf16 only; fp16 is disabled because loss "
            "scaling is not implemented"
        )

    if requested_device == "auto":
        device_type = "cuda" if cuda_available else "npu" if npu_available else "cpu"
    else:
        device_type = requested_device

    if device_type == "cuda" and not cuda_available:
        raise RuntimeError("Stage B was asked to use CUDA, but CUDA is unavailable")
    if device_type == "npu" and not npu_available:
        raise RuntimeError("Stage B was asked to use an Ascend NPU, but NPU is unavailable")

    if device_type == "cuda":
        if world_size < 4:
            raise RuntimeError(
                "Stage B CUDA full-parameter CID training requires at least four GPU ranks"
            )
        return "cuda", local_rank, distributed_backend("cuda")

    if device_type == "npu":
        if world_size in {2, 3}:
            raise RuntimeError(
                "Stage B NPU training supports one NPU rank for compact models or "
                "at least four NPU ranks for sharded large-model training"
            )
        if cpu_offload:
            raise ValueError("--fsdp-cpu-offload is not used by NPU Stage B training")
        return "npu", local_rank, distributed_backend("npu")

    if cpu_offload:
        raise ValueError("--fsdp-cpu-offload is only meaningful for CUDA Stage B training")
    return "cpu", None, distributed_backend("cpu")


def _init_stage_b_process_group(
    dist,
    *,
    backend: str,
    world_size: int,
) -> Path | None:
    """Initialize torch.distributed, including direct single-process CPU launches."""
    if dist.is_initialized():
        return None
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend)
        return None
    if world_size != 1:
        raise RuntimeError("multi-rank Stage B must be launched with torchrun")

    rendezvous_dir = Path(tempfile.mkdtemp(prefix="cid-stage-b-rdzv-"))
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{rendezvous_dir / 'init'}",
        rank=0,
        world_size=1,
    )
    return rendezvous_dir

def _train_stage_b(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from cid.model import (
        CIDTrainer,
        CIDTrainerConfig,
        ILLaDACIDAdapter,
        ILLaDACIDConfig,
        ILLaDATrajectoryTensorizer,
        balance_rollout_windows_by_semantic_task,
        load_cid_adapter_checkpoint,
        load_cid_adapter_from_pretrained,
        load_stage_b_checkpoint,
        materialize_indexed_rollout_windows,
        save_stage_b_checkpoint,
        shard_rollout_windows,
        stage_b_adamw_parameter_groups,
        stage_b_consumed_windows_by_bucket,
        stage_b_gradient_accumulation_steps,
        stage_b_optimizer_steps_per_epoch,
        trajectory_rollout_windows,
        wrap_stage_b_fsdp,
    )
    from cid.model.encoding import ILLaDATextEncoder
    from cid.model.loading import backbone_model_type, pretrained_revision

    if args.thought_capacity != 128:
        raise ValueError("CID v1 Stage B requires --thought-capacity 128")
    if args.resume and args.init_cid_checkpoint:
        raise ValueError("--resume and --init-cid-checkpoint are mutually exclusive")
    if not args.resume and not args.init_cid_checkpoint:
        raise ValueError("Stage B requires --init-cid-checkpoint unless --resume is used")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be positive")
    if args.target_global_batch_size <= 0:
        raise ValueError("--target-global-batch-size must be positive")
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.mlp_chunk_size <= 0 or args.norm_chunk_size <= 0:
        raise ValueError("Stage B chunk sizes must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.min_learning_rate_ratio <= 1.0:
        raise ValueError("--min-learning-rate-ratio must be in [0, 1]")
    if args.backbone_lr_scale <= 0.0:
        raise ValueError("--backbone-lr-scale must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_type, device_index, backend = _stage_b_execution_target(
        args.device,
        cuda_available=torch_device_available(torch, "cuda"),
        npu_available=torch_device_available(torch, "npu"),
        world_size=world_size,
        local_rank=local_rank,
        dtype=args.dtype,
        cpu_offload=args.fsdp_cpu_offload,
    )

    gradient_accumulation_steps = stage_b_gradient_accumulation_steps(
        world_size=world_size,
        micro_batch_size=args.micro_batch_size,
        target_global_batch_size=args.target_global_batch_size,
        explicit_steps=args.gradient_accumulation_steps,
    )
    effective_batch = (
        args.micro_batch_size * gradient_accumulation_steps * world_size
    )

    compute_dtype = torch.bfloat16
    if device_type in {"cuda", "npu"}:
        assert device_index is not None
        configure_torch_accelerator(torch, device_type, device_index)
        device = torch.device(device_type, device_index)
    else:
        device = torch.device("cpu")
    single_npu_stage_b = device_type == "npu" and world_size == 1
    if single_npu_stage_b and backbone_model_type(args.model) != "lfm2":
        raise RuntimeError(
            "single-NPU Stage B is supported only for the compact LFM2 CID-v1-0.4B backbone; "
            "larger backbones require at least four NPU ranks"
        )
    rendezvous_dir = _init_stage_b_process_group(
        dist,
        backend=backend,
        world_size=world_size,
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        adapter_config = ILLaDACIDConfig(
            max_thought_slots=args.thought_capacity,
            max_display_tokens=args.max_display_tokens,
            display_canvas_tokens=args.display_canvas_tokens,
        )
        dataset_manifest = inspect_dataset(args.data)
        validate_neural_training_contract(
            dataset_manifest,
            max_argument_slots=adapter_config.max_argument_slots,
            max_need_slots=adapter_config.max_need_slots,
        )
        if dataset_manifest.thought_capacity_required > args.thought_capacity:
            raise ValueError(
                "training data requires a larger TCT capacity: "
                f"{dataset_manifest.thought_capacity_required} > {args.thought_capacity}"
            )

        examples, validation_examples = _index_train_and_load_validation_examples(
            args.data,
            validation_data_path=args.validation_data,
            max_examples=args.max_examples,
            max_validation_examples=args.max_validation_examples,
        )
        windows = balance_rollout_windows_by_semantic_task(
            trajectory_rollout_windows(examples, max_horizon=args.rollout_horizon)
        )
        if not windows:
            raise ValueError("training data contains no adjacent thought transitions")
        validation_windows = balance_rollout_windows_by_semantic_task(
            trajectory_rollout_windows(
                validation_examples,
                max_horizon=args.rollout_horizon,
            )
        )
        transition_count_total = sum(len(window.source_steps) for window in windows)
        validation_transition_count_total = sum(
            len(window.source_steps) for window in validation_windows
        )
        local_validation_windows = (
            shard_rollout_windows(
                validation_windows,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                epoch=0,
                shuffle=False,
                micro_batch_size=args.micro_batch_size,
                portable_bucket_order=True,
            )
            if validation_windows
            else ()
        )

        resume_rank0_state = None
        if args.resume:
            resume_path = Path(args.resume)
            resume_state_path = (
                resume_path if single_npu_stage_b else resume_path / "rank-0000.pt"
            )
            resume_rank0_state = torch.load(
                resume_state_path,
                map_location="cpu",
                weights_only=False,
            )

        lr_decay_steps = max(
            1,
            sum(
                stage_b_optimizer_steps_per_epoch(
                    windows,
                    world_size=world_size,
                    micro_batch_size=args.micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    seed=args.seed,
                    epoch=epoch,
                    shuffle=not args.no_shuffle,
                    length_aware=True,
                    portable_bucket_order=True,
                )
                for epoch in range(1, args.epochs + 1)
            ),
        )
        warmup_steps = (
            max(1, round(lr_decay_steps * args.warmup_ratio))
            if args.warmup_ratio > 0.0
            else 0
        )
        if resume_rank0_state is not None:
            saved_trainer_config = resume_rank0_state["trainer_config"]
            lr_decay_steps = int(saved_trainer_config["lr_decay_steps"])
            warmup_steps = int(saved_trainer_config["warmup_steps"])

        tokenizer_kwargs: dict[str, object] = {"trust_remote_code": True}
        revision = pretrained_revision(args.model)
        if revision is not None:
            tokenizer_kwargs["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)

        def load_adapter() -> tuple[ILLaDACIDAdapter, ILLaDATextEncoder]:
            # Keep the initial FP32 model on host memory. On CUDA, FSDP's device_id moves
            # each wrap unit onto the local GPU while sharding it; on CPU it shards in place.
            model = load_cid_adapter_from_pretrained(
                args.model,
                config=adapter_config,
                freeze_backbone=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            if args.init_cid_checkpoint:
                load_cid_adapter_checkpoint(
                    model,
                    args.init_cid_checkpoint,
                    expected_semantic_pooling=args.semantic_pooling,
                )
            snapshot = ILLaDATextEncoder.from_frozen_snapshot(
                model,
                tokenizer,
                device=device,
                dtype=compute_dtype,
                embedding_device="cpu",
                pooling_mode=args.semantic_pooling,
            )
            # Ascend ranks have ample device memory but commonly share a tighter host-memory
            # cgroup. Move each freshly loaded model to its NPU before the next rank loads.
            if device_type == "npu":
                model.set_device_value_validation(False)
                model = model.to(device)
            model.set_backbone_trainable(True)
            if args.gradient_checkpointing:
                model.set_gradient_checkpointing(True, use_reentrant=False)
            model.set_mlp_chunk_size(args.mlp_chunk_size)
            model.set_norm_chunk_size(args.norm_chunk_size)
            return model, snapshot

        adapter = None
        text_encoder = None
        for loading_rank in range(world_size):
            if rank == loading_rank:
                adapter, text_encoder = load_adapter()
            dist.barrier()
        if adapter is None or text_encoder is None:
            raise RuntimeError("failed to load Stage B CID model on this training rank")

        optimizer_groups = stage_b_adamw_parameter_groups(
            adapter,
            backbone_lr_scale=args.backbone_lr_scale,
            weight_decay=args.weight_decay,
        )
        if single_npu_stage_b:
            training_model = wrap_npu_autocast(torch, adapter, dtype=compute_dtype)
            gradient_clipper = None
        else:
            training_model = wrap_stage_b_fsdp(
                adapter,
                device_id=device,
                compute_dtype=compute_dtype,
                cpu_offload=args.fsdp_cpu_offload,
            )
            gradient_clipper = training_model.clip_grad_norm_
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=False,
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
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                warmup_steps=warmup_steps,
                lr_decay_steps=lr_decay_steps,
                min_learning_rate_ratio=args.min_learning_rate_ratio,
                timestep_min=args.timestep_min,
                timestep_max=args.timestep_max,
                rollout_horizon=args.rollout_horizon,
                rollout_allocation_threshold=args.rollout_allocation_threshold,
                rollout_max_allocations_per_step=args.rollout_max_allocations_per_step,
                teacher_forcing_epochs=args.teacher_forcing_epochs,
                rollout_ramp_epochs=args.rollout_ramp_epochs,
                semantic_pooling=args.semantic_pooling,
                seed=args.seed,
            ),
            optimizer=optimizer,
            forward_model=training_model,
            gradient_clipper=gradient_clipper,
        )
        loaded_checkpoint_metadata = None
        if args.resume:
            if single_npu_stage_b:
                trainer.load_checkpoint(
                    args.resume,
                    expected_dataset_sha256=dataset_manifest.sha256,
                )
            else:
                loaded_checkpoint_metadata = load_stage_b_checkpoint(
                    training_model,
                    optimizer,
                    trainer,
                    args.resume,
                    expected_dataset_sha256=dataset_manifest.sha256,
                )
        else:
            trainer.reseed(args.seed + rank)

        saved_world_size = (
            int(loaded_checkpoint_metadata["world_size"])
            if loaded_checkpoint_metadata is not None
            else world_size
        )
        world_size_changed = saved_world_size != world_size
        saved_partial_windows = (
            int(resume_rank0_state["trainer_state"].get("rollout_windows_seen_in_epoch", 0))
            if resume_rank0_state is not None
            else 0
        )
        resume_epoch_progress = (
            loaded_checkpoint_metadata.get("epoch_progress")
            if loaded_checkpoint_metadata is not None
            else None
        )
        if world_size_changed and saved_partial_windows and trainer.data_order_version < 4:
            raise ValueError(
                "cross-world-size Stage B mid-epoch resume requires data-order v4; "
                "resume this legacy checkpoint with its original world size"
            )
        if world_size_changed and saved_partial_windows and resume_epoch_progress is None:
            raise ValueError(
                "cross-world-size Stage B resume requires a v3 partial-epoch data cursor"
            )

        output_dir = Path(args.output_dir)
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"stage=B device={device} world_size={world_size} dtype={args.dtype} "
                f"optimizer=adamw examples={len(examples)} transitions={transition_count_total} "
                f"validation_examples={len(validation_examples)} "
                f"validation_transitions={validation_transition_count_total} "
                f"target_global_batch={args.target_global_batch_size} "
                f"effective_batch={effective_batch} grad_accum={gradient_accumulation_steps} "
                f"mlp_chunk={args.mlp_chunk_size} norm_chunk={args.norm_chunk_size} "
                f"peak_cid_lr={args.learning_rate:.3e} "
                f"peak_backbone_lr={args.learning_rate * args.backbone_lr_scale:.3e} "
                f"warmup_steps={warmup_steps} lr_decay_steps={lr_decay_steps} "
                f"target_epochs={args.epochs}",
                flush=True,
            )
        dist.barrier()

        if args.log_every_steps <= 0:
            raise ValueError("--log-every-steps must be positive")
        if args.checkpoint_every_steps <= 0:
            raise ValueError("--checkpoint-every-steps must be positive")
        metrics_path = output_dir / f"train_metrics.rank-{rank:04d}.jsonl"
        validation_metrics_path = output_dir / "validation_metrics.jsonl"
        run_started = time.monotonic()
        next_checkpoint_step = (
            trainer.state.optimizer_steps // args.checkpoint_every_steps + 1
        ) * args.checkpoint_every_steps
        # Only periodic checkpoints created in this process are disposable. A resume
        # checkpoint may be a persistent epoch snapshot and must never be deleted.
        last_periodic_checkpoint = None

        if trainer.state.epochs_completed >= args.epochs:
            if rank == 0:
                print(
                    f"Stage B target already satisfied: "
                    f"epochs_completed={trainer.state.epochs_completed} target={args.epochs}",
                    flush=True,
                )
            return

        first_epoch = trainer.state.epochs_completed + 1
        for epoch in range(first_epoch, args.epochs + 1):
            epoch_base_consumed: dict[str, int] = {}
            if epoch == first_epoch and resume_epoch_progress is not None:
                if int(resume_epoch_progress.get("epoch", epoch)) != epoch:
                    raise ValueError("Stage B checkpoint epoch cursor does not match trainer state")
                cursor_name = (
                    "consumed_by_bucket" if world_size_changed else "base_consumed_by_bucket"
                )
                epoch_base_consumed = {
                    str(key): int(value)
                    for key, value in resume_epoch_progress.get(cursor_name, {}).items()
                }
            epoch_shard = shard_rollout_windows(
                windows,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                epoch=epoch,
                shuffle=not args.no_shuffle,
                micro_batch_size=args.micro_batch_size,
                consumed_windows_by_bucket=epoch_base_consumed,
                length_aware=trainer.data_order_version >= 2,
                zero_gradient_padding=trainer.data_order_version >= 3,
                portable_bucket_order=trainer.data_order_version >= 4,
            )
            total_local_windows = len(epoch_shard)
            resumed_windows = trainer.state.rollout_windows_seen_in_epoch
            if resumed_windows:
                if resumed_windows >= total_local_windows:
                    raise ValueError(
                        "checkpoint rollout position is outside the current epoch shard"
                    )
                if rank == 0:
                    print(
                        f"resume stage=B epoch={epoch} "
                        f"optimizer_steps={trainer.state.optimizer_steps} "
                        f"windows_seen={resumed_windows}/{total_local_windows}",
                        flush=True,
                    )
            local_windows = epoch_shard[resumed_windows:]
            if world_size_changed and rank == 0 and epoch == first_epoch:
                print(
                    f"elastic-resume stage=B old_world_size={saved_world_size} "
                    f"new_world_size={world_size} optimizer_steps={trainer.state.optimizer_steps}",
                    flush=True,
                )
            local_windows = materialize_indexed_rollout_windows(args.data, local_windows)
            rollout_probability = trainer.rollout_probability()

            def report_progress(
                progress,
                *,
                current_epoch: int = epoch,
                current_rollout_probability: float = rollout_probability,
                current_total_windows: int = total_local_windows,
                current_epoch_shard=epoch_shard,
                current_base_consumed=epoch_base_consumed,
            ) -> None:
                nonlocal next_checkpoint_step, last_periodic_checkpoint
                group_lrs = {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                }
                backbone_lrs = [
                    lr for name, lr in group_lrs.items() if name.startswith("backbone-")
                ]
                cid_lrs = [lr for name, lr in group_lrs.items() if name.startswith("cid-")]
                record = {
                    "timestamp": time.time(),
                    "elapsed_seconds": time.monotonic() - run_started,
                    "epoch": current_epoch,
                    "rank": rank,
                    "world_size": world_size,
                    "optimizer_steps": progress.optimizer_steps,
                    "interval_transitions": progress.transitions,
                    "mean_loss": progress.mean_loss,
                    "raw_mean_loss": progress.raw_mean_loss,
                    "component_losses": progress.component_mean_losses,
                    "learning_rate": progress.learning_rate,
                    "backbone_learning_rate": min(backbone_lrs) if backbone_lrs else None,
                    "cid_learning_rate": max(cid_lrs) if cid_lrs else None,
                    "rollout_probability": current_rollout_probability,
                    "windows_seen_in_epoch": progress.rollout_windows_seen_in_epoch,
                    "windows_total_in_epoch": current_total_windows,
                    "effective_batch": effective_batch,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if rank == 0:
                    print(
                        f"progress stage=B epoch={current_epoch} "
                        f"optimizer_steps={progress.optimizer_steps} "
                        f"loss={progress.mean_loss:.6f} raw_loss={progress.raw_mean_loss:.6f} "
                        f"windows={progress.rollout_windows_seen_in_epoch}/{current_total_windows}",
                        flush=True,
                    )

                checkpoint_due = progress.optimizer_steps >= next_checkpoint_step
                if (
                    not checkpoint_due
                    or progress.rollout_windows_seen_in_epoch >= current_total_windows
                ):
                    return

                clean = torch.tensor(
                    [1 if trainer.pending_accumulation_steps == 0 else 0],
                    device=device,
                    dtype=torch.int32,
                )
                dist.all_reduce(clean, op=dist.ReduceOp.MIN)
                if int(clean.item()) == 0:
                    return

                checkpoint = output_dir / (
                    f"stage-b-step-{progress.optimizer_steps:08d}.pt"
                    if single_npu_stage_b
                    else f"stage-b-step-{progress.optimizer_steps:08d}"
                )
                if single_npu_stage_b:
                    trainer.save_checkpoint(
                        checkpoint,
                        dataset_sha256=dataset_manifest.sha256,
                    )
                else:
                    consumed_by_bucket = stage_b_consumed_windows_by_bucket(
                        windows,
                        current_epoch_shard,
                        local_windows_seen=progress.rollout_windows_seen_in_epoch,
                        world_size=world_size,
                        base_consumed_by_bucket=current_base_consumed,
                    )
                    save_stage_b_checkpoint(
                        training_model,
                        optimizer,
                        trainer,
                        checkpoint,
                        dataset_sha256=dataset_manifest.sha256,
                        epoch_progress={
                            "epoch": current_epoch,
                            "base_consumed_by_bucket": current_base_consumed,
                            "consumed_by_bucket": consumed_by_bucket,
                        },
                    )
                previous = last_periodic_checkpoint
                last_periodic_checkpoint = checkpoint
                if rank == 0:
                    latest = output_dir / (
                        "stage-b-latest.pt" if single_npu_stage_b else "stage-b-latest"
                    )
                    latest.unlink(missing_ok=True)
                    latest.symlink_to(
                        checkpoint.name, target_is_directory=not single_npu_stage_b
                    )
                    if previous is not None and previous != checkpoint:
                        if single_npu_stage_b:
                            previous.unlink(missing_ok=True)
                        else:
                            shutil.rmtree(previous, ignore_errors=True)
                    print(
                        f"checkpoint stage=B optimizer_steps={progress.optimizer_steps} "
                        f"path={checkpoint}",
                        flush=True,
                    )
                while next_checkpoint_step <= progress.optimizer_steps:
                    next_checkpoint_step += args.checkpoint_every_steps

            report = trainer.train_rollout_windows(
                local_windows,
                epochs=1,
                shuffle=False,
                preserve_order=True,
                progress_every_optimizer_steps=args.log_every_steps,
                progress_callback=report_progress,
            )
            aggregate = torch.tensor(
                [
                    report.mean_loss * report.transitions,
                    report.raw_mean_loss * report.transitions,
                    float(report.transitions),
                ],
                device=device,
                dtype=torch.float32 if device_type == "npu" else torch.float64,
            )
            dist.all_reduce(aggregate)
            mean_loss = float(aggregate[0] / aggregate[2])
            raw_mean_loss = float(aggregate[1] / aggregate[2])
            transition_count = int(aggregate[2])

            checkpoint = output_dir / (
                f"stage-b-epoch-{epoch:04d}.pt"
                if single_npu_stage_b
                else f"stage-b-epoch-{epoch:04d}"
            )
            if single_npu_stage_b:
                trainer.save_checkpoint(
                    checkpoint,
                    dataset_sha256=dataset_manifest.sha256,
                )
            else:
                save_stage_b_checkpoint(
                    training_model,
                    optimizer,
                    trainer,
                    checkpoint,
                    dataset_sha256=dataset_manifest.sha256,
                )
            if rank == 0:
                step_alias = output_dir / (
                    f"stage-b-step-{trainer.state.optimizer_steps:08d}.pt"
                    if single_npu_stage_b
                    else f"stage-b-step-{trainer.state.optimizer_steps:08d}"
                )
                _replace_checkpoint_alias(
                    step_alias,
                    checkpoint,
                    target_is_directory=not single_npu_stage_b,
                )
                _replace_checkpoint_alias(
                    output_dir / (
                        "stage-b-latest.pt" if single_npu_stage_b else "stage-b-latest"
                    ),
                    checkpoint,
                    target_is_directory=not single_npu_stage_b,
                )
                if last_periodic_checkpoint is not None:
                    if single_npu_stage_b:
                        last_periodic_checkpoint.unlink(missing_ok=True)
                    else:
                        shutil.rmtree(last_periodic_checkpoint, ignore_errors=True)
                    last_periodic_checkpoint = None
            dist.barrier()

            validation_mean_loss = None
            if local_validation_windows:
                teacher_forced_report = trainer.evaluate_rollout_windows(
                    local_validation_windows,
                    seed=args.seed + 1_000_003,
                    rollout_probability=0.0,
                )
                teacher_forced = _aggregate_validation_report(
                    teacher_forced_report,
                    torch_module=torch,
                    dist_module=dist,
                    device=device,
                    device_type=device_type,
                    distributed=True,
                )
                free_rollout_report = trainer.evaluate_rollout_windows(
                    local_validation_windows,
                    seed=args.seed + 1_000_003,
                    rollout_probability=1.0,
                )
                free_rollout = _aggregate_validation_report(
                    free_rollout_report,
                    torch_module=torch,
                    dist_module=dist,
                    device=device,
                    device_type=device_type,
                    distributed=True,
                )
                validation_mean_loss = float(teacher_forced["mean_loss"])
                validation_raw_mean_loss = float(teacher_forced["raw_mean_loss"])
                validation_transitions = int(teacher_forced["transitions"])
                if rank == 0:
                    validation_record = {
                        "timestamp": time.time(),
                        "elapsed_seconds": time.monotonic() - run_started,
                        "epoch": epoch,
                        "optimizer_steps": trainer.state.optimizer_steps,
                        "validation_examples": len(validation_examples),
                        "validation_transitions": validation_transitions,
                        "validation_mean_loss": validation_mean_loss,
                        "validation_raw_mean_loss": validation_raw_mean_loss,
                        "validation_seed": args.seed + 1_000_003,
                        "objective": "teacher_forced_and_free_rollout_fixed_noise",
                        "teacher_forced": teacher_forced,
                        "free_rollout": free_rollout,
                        "checkpoint": str(checkpoint),
                    }
                    with validation_metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(validation_record, sort_keys=True) + "\n")

            if rank == 0:
                validation_text = (
                    "disabled"
                    if validation_mean_loss is None
                    else f"{validation_mean_loss:.6f}"
                )
                print(
                    f"epoch={epoch} transitions={transition_count} "
                    f"optimizer_steps={trainer.state.optimizer_steps} mean_loss={mean_loss:.6f} "
                    f"raw_mean_loss={raw_mean_loss:.6f} validation_loss={validation_text} "
                    f"rollout_probability={rollout_probability:.3f} checkpoint={checkpoint}",
                    flush=True,
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous_dir is not None:
            shutil.rmtree(rendezvous_dir, ignore_errors=True)


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
    self_identity.add_argument("--contract", default="configs/cid-self-identity-v1.contract.json")
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
    multilingual.add_argument("--output-dir", default="data/generated/multilingual-v1")
    multilingual.add_argument("--zh-tasks", type=int, default=450)
    multilingual.add_argument("--en-zh-tasks", type=int, default=300)
    multilingual.add_argument("--ja-tasks", type=int, default=225)
    multilingual.add_argument("--es-tasks", type=int, default=225)
    multilingual.add_argument("--schedule-variants", type=int, default=2)
    multilingual.add_argument("--seed", type=int, default=20260813)
    composed = subparsers.add_parser(
        "build-composed-training",
        help="build mixed lookup/reasoning/calculator/symbolic CID trajectories",
    )
    composed.add_argument("--output-dir", default="data/generated")
    composed.add_argument(
        "--reference-manifest-output",
        default="data/composed-teacher-v1.reference-manifest.json",
    )
    composed.add_argument("--count-per-family", type=int, default=2000)
    composed.add_argument("--variants-per-task", type=int, default=2)
    composed.add_argument("--thought-capacity", type=int, default=8)
    composed.add_argument("--min-delay-steps", type=int, default=1)
    composed.add_argument("--max-delay-steps", type=int, default=4)
    composed.add_argument("--seed", type=int, default=20260813)
    restraint = subparsers.add_parser(
        "build-tool-restraint-training",
        help="derive natural tasks where tools are available but unnecessary",
    )
    restraint.add_argument(
        "--source-tasks", default="data/generated/public-teacher-tasks-v1.train.jsonl"
    )
    restraint.add_argument(
        "--source-plans", default="data/generated/public-teacher-plans-v1.accepted.jsonl"
    )
    restraint.add_argument("--output-dir", default="data/generated")
    restraint.add_argument(
        "--reference-manifest-output",
        default="data/tool-restraint-teacher-v1.reference-manifest.json",
    )
    restraint.add_argument("--count", type=int, default=6500)
    restraint.add_argument("--thought-capacity", type=int, default=8)
    restraint.add_argument("--seed", type=int, default=20260813)
    deep_restraint = subparsers.add_parser(
        "build-deep-tool-restraint-training",
        help="derive deep long-tail tasks where exposed external tools should remain unused",
    )
    deep_restraint.add_argument(
        "--source-tasks", default="data/generated/compositional-teacher-tasks-v1.jsonl"
    )
    deep_restraint.add_argument(
        "--source-plans",
        default="data/generated/compositional-teacher-plans-v1.accepted.jsonl",
    )
    deep_restraint.add_argument("--output-dir", default="data/generated/deep-tool-restraint-v1")
    deep_restraint.add_argument(
        "--reference-manifest-output",
        default="data/deep-tool-restraint-v1.reference-manifest.json",
    )
    deep_restraint.add_argument("--count-per-bucket", type=int, default=1000)
    deep_restraint.add_argument("--min-dependency-depth", type=int, default=8)
    deep_restraint.add_argument(
        "--capacity-buckets", type=int, nargs="+", default=[16, 32, 64, 128]
    )
    deep_restraint.add_argument("--seed", type=int, default=20260813)

    natural_interaction = subparsers.add_parser(
        "build-natural-interaction-training",
        help="derive grounded long-form natural retrieval trajectories with varied tool schemas",
    )
    natural_interaction.add_argument(
        "--public-tasks", default="data/generated/public-teacher-tasks-v1.train.jsonl"
    )
    natural_interaction.add_argument(
        "--public-plans", default="data/generated/public-teacher-plans-v1.accepted.jsonl"
    )
    natural_interaction.add_argument(
        "--interaction-tasks",
        default="data/generated/public-interaction-teacher-tasks-v1.train.jsonl",
    )
    natural_interaction.add_argument(
        "--interaction-plans",
        default="data/generated/public-interaction-teacher-plans-v1.accepted.jsonl",
    )
    natural_interaction.add_argument(
        "--output-dir", default="data/generated/natural-interaction-v1"
    )
    natural_interaction.add_argument(
        "--reference-manifest-output",
        default="data/natural-interaction-v1.reference-manifest.json",
    )
    natural_interaction.add_argument("--variants-per-task", type=int, default=2)
    natural_interaction.add_argument("--thought-capacity", type=int, default=8)
    natural_interaction.add_argument("--min-delay-steps", type=int, default=1)
    natural_interaction.add_argument("--max-delay-steps", type=int, default=4)
    natural_interaction.add_argument("--seed", type=int, default=20260813)

    natural_public = subparsers.add_parser(
        "build-natural-public-training",
        help="build train-only natural public CID tasks and grounded trajectories",
    )
    natural_public.add_argument(
        "--source",
        required=True,
        choices=("nq-open", "oasst1", "multidoc2dial", "qasper"),
    )
    natural_public.add_argument("--registry", default="configs/natural-public-datasets-v1.json")
    natural_public.add_argument("--output-dir", default=None)
    natural_public.add_argument("--reference-manifest-output", default=None)
    natural_public.add_argument("--exclude-trajectories", nargs="*", default=[])
    natural_public.add_argument("--cache-dir", default=None)
    natural_public.add_argument("--variants-per-task", type=int, default=2)
    natural_public.add_argument("--thought-capacity", type=int, default=8)
    natural_public.add_argument("--min-delay-steps", type=int, default=1)
    natural_public.add_argument("--max-delay-steps", type=int, default=4)
    natural_public.add_argument("--seed", type=int, default=20260814)

    surface = subparsers.add_parser(
        "build-surface-diverse-training",
        help="build surface-diversified replacements for template-heavy training components",
    )
    surface.add_argument(
        "--component",
        choices=(
            "composed",
            "long-horizon",
            "mechanism-v3",
            "computational-v3",
            "symbolic-v3",
            "correction-v3",
            "composed-v3",
            "long-horizon-v3",
            "logic-v3",
            "self-identity-v3",
            "multilingual-v3",
            "compositional-v3",
            "deep-restraint-v3",
            "logic-v4",
            "compositional-v4",
            "deep-restraint-v4",
        ),
        required=True,
    )
    surface.add_argument("--max-tasks", type=int)
    surface.add_argument("--seed", type=int, default=20260813)

    long_horizon = subparsers.add_parser(
        "build-long-horizon-training",
        help="build depth-4-to-6 dependent asynchronous read-only tool trajectories",
    )
    long_horizon.add_argument("--output-dir", default="data/generated/long-horizon-v1")
    long_horizon.add_argument(
        "--reference-manifest-output",
        default="data/long-horizon-teacher-v1.reference-manifest.json",
    )
    long_horizon.add_argument("--count-per-family", type=int, default=2000)
    long_horizon.add_argument("--variants-per-task", type=int, default=2)
    long_horizon.add_argument("--thought-capacity", type=int, default=12)
    long_horizon.add_argument("--min-delay-steps", type=int, default=1)
    long_horizon.add_argument("--max-delay-steps", type=int, default=5)
    long_horizon.add_argument("--seed", type=int, default=20260813)

    compositional = subparsers.add_parser(
        "build-compositional-training",
        help="build 8/16/32/64/128-slot compositional long-tail reasoning data and OOD probes",
    )
    compositional.add_argument("--output-dir", default="data/generated")
    compositional.add_argument("--seed", type=int, default=20260813)
    compositional.add_argument("--variants-per-task", type=int, default=2)
    compositional.add_argument("--probe-variants-per-task", type=int, default=1)

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
    migrate_v3 = subparsers.add_parser(
        "migrate-dataset-contract-v3",
        help="stream a trajectory JSONL into the neural-contract-v3 data ABI",
    )
    migrate_v3.add_argument("--data", required=True)
    migrate_v3.add_argument("--output", required=True)
    migrate_v3.add_argument("--manifest-output", required=True)
    validation_v3 = subparsers.add_parser(
        "build-contract-v3-validation",
        help="build a validation mix with OOD reasoning plus held-out synthetic tool interactions",
    )
    validation_v3.add_argument("--reasoning-source", required=True)
    validation_v3.add_argument("--output", required=True)
    validation_v3.add_argument("--manifest-output", required=True)
    validation_v3.add_argument("--examples", type=int, default=512)
    validation_v3.add_argument("--tool-examples", type=int, default=96)
    validation_v3.add_argument("--seed", type=int, default=20260829)
    migrate_v4 = subparsers.add_parser(
        "migrate-dataset-contract-v4",
        help="rematerialize trajectories for continuous answer-draft Display supervision",
    )
    migrate_v4.add_argument("--data", required=True)
    migrate_v4.add_argument("--output", required=True)
    migrate_v4.add_argument("--manifest-output", required=True)
    migrate_v4.add_argument("--curated-data")
    validation_v4 = subparsers.add_parser(
        "build-contract-v4-validation",
        help="build v4 OOD reasoning and held-out tool validation with Display audits",
    )
    validation_v4.add_argument("--reasoning-source", required=True)
    validation_v4.add_argument("--output", required=True)
    validation_v4.add_argument("--manifest-output", required=True)
    validation_v4.add_argument("--examples", type=int, default=512)
    validation_v4.add_argument("--tool-examples", type=int, default=96)
    validation_v4.add_argument("--curated-examples", type=int, default=48)
    validation_v4.add_argument("--seed", type=int, default=20260903)
    curated_v4 = subparsers.add_parser(
        "build-curated-v4-training",
        help="build the hand-authored v4 Display/TCT curriculum",
    )
    curated_v4.add_argument(
        "--output", default="data/generated/curated-v4/curated-v4-trajectories.jsonl"
    )
    curated_v4.add_argument(
        "--manifest-output",
        default="data/generated/curated-v4/curated-v4-trajectories.manifest.json",
    )
    curated_v4.add_argument("--count-per-family", type=int, default=48)
    curated_v4.add_argument("--seed", type=int, default=20260903)
    curated_v4.add_argument("--thought-capacity", type=int, default=8)
    curated_v4.add_argument("--training-weight", type=float, default=4.0)
    trajectory_mixture = subparsers.add_parser(
        "materialize-trajectory-mixture",
        help="verify and concatenate pinned CID trajectory components into one training JSONL",
    )
    trajectory_mixture.add_argument("--spec", required=True)
    trajectory_mixture.add_argument("--output", required=True)
    trajectory_mixture.add_argument("--manifest-output", required=True)
    training_audit = subparsers.add_parser(
        "audit-training-data",
        help="audit actual trainable transition mass and causal first-need invariants",
    )
    training_audit.add_argument("--data", required=True)
    training_audit.add_argument("--strict", action="store_true")
    public_pool = subparsers.add_parser(
        "build-public-task-pool",
        help="build the pinned public semantic-task pool from registered training splits",
    )
    public_pool.add_argument("--registry", default="configs/public-datasets.json")
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
    benchmark.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    benchmark.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    policy_tuning = benchmark.add_argument_group("neural policy tuning")
    policy_tuning.add_argument("--denoising-steps", type=int, default=8)
    policy_tuning.add_argument(
        "--display-revision-fraction",
        type=float,
        default=DEFAULT_DISPLAY_REVISION_FRACTION,
    )
    policy_tuning.add_argument(
        "--display-revision-margin",
        type=float,
        default=DEFAULT_DISPLAY_REVISION_MARGIN,
    )
    materializer_tuning = benchmark.add_argument_group("materialization tuning")
    materializer_tuning.add_argument(
        "--allocation-threshold", type=float, default=DEFAULT_ALLOCATION_THRESHOLD
    )
    materializer_tuning.add_argument(
        "--convergence-threshold", type=float, default=DEFAULT_CONVERGENCE_THRESHOLD
    )
    materializer_tuning.add_argument("--need-threshold", type=float, default=DEFAULT_NEED_THRESHOLD)
    materializer_tuning.add_argument(
        "--need-target-cell-threshold",
        type=float,
        default=DEFAULT_NEED_TARGET_CELL_THRESHOLD,
    )
    materializer_tuning.add_argument(
        "--need-target-display-threshold",
        type=float,
        default=DEFAULT_NEED_TARGET_DISPLAY_THRESHOLD,
    )
    materializer_tuning.add_argument(
        "--argument-presence-threshold",
        type=float,
        default=DEFAULT_ARGUMENT_PRESENCE_THRESHOLD,
    )
    materializer_tuning.add_argument(
        "--anchor-presence-threshold",
        type=float,
        default=DEFAULT_ANCHOR_PRESENCE_THRESHOLD,
    )
    materializer_tuning.add_argument(
        "--link-presence-threshold",
        type=float,
        default=DEFAULT_LINK_PRESENCE_THRESHOLD,
    )
    materializer_tuning.add_argument(
        "--retrieval-similarity-threshold",
        type=float,
        default=DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
    )
    materializer_tuning.add_argument(
        "--max-allocations-per-step",
        type=int,
        default=DEFAULT_MAX_ALLOCATIONS_PER_STEP,
    )
    materializer_tuning.add_argument(
        "--max-age-s", type=float, default=DEFAULT_MATERIALIZED_MAX_AGE_S
    )
    runtime_defaults = RuntimeConfig()
    runtime_tuning = benchmark.add_argument_group("runtime tuning")
    runtime_tuning.add_argument("--max-steps", type=int, default=32)
    runtime_tuning.add_argument(
        "--max-wall-time-s", type=float, default=runtime_defaults.max_wall_time_s
    )
    runtime_tuning.add_argument(
        "--binding-threshold", type=float, default=DEFAULT_BINDING_THRESHOLD
    )
    runtime_tuning.add_argument("--idle-yield-s", type=float, default=runtime_defaults.idle_yield_s)
    runtime_tuning.add_argument(
        "--reclamation-grace-steps",
        type=int,
        default=runtime_defaults.reclamation_grace_steps,
    )
    runtime_tuning.add_argument(
        "--reclamation-low-watermark",
        type=float,
        default=runtime_defaults.reclamation_low_watermark,
    )
    runtime_tuning.add_argument(
        "--reclamation-target-watermark",
        type=float,
        default=runtime_defaults.reclamation_target_watermark,
    )
    benchmark.add_argument("--max-examples", type=int)
    benchmark.add_argument("--progress-every", type=int, default=10)
    benchmark.add_argument("--seed-teacher-state", action="store_true")
    train = subparsers.add_parser(
        "train",
        help="run Stage A CID adapter training with a frozen diffusion backbone",
    )
    train.add_argument("--data", required=True)
    train.add_argument(
        "--validation-data",
        help=(
            "optional held-out trajectory JSONL; when omitted, validation examples are "
            "taken from --data entries with metadata.split=validation"
        ),
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--model", default="GSAI-ML/iLLaDA-8B-Base")
    train.add_argument("--resume")
    train.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    train.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--micro-batch-size", type=int, default=1)
    train.add_argument(
        "--physical-micro-batch-size",
        type=int,
        help=(
            "execution-only Stage A split that lowers peak memory while preserving logical "
            "micro-batch and checkpoint geometry"
        ),
    )
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--warmup-steps", type=int, default=0)
    train.add_argument("--lr-decay-steps", type=int, default=0)
    train.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    train.add_argument("--timestep-min", type=float, default=0.05)
    train.add_argument("--timestep-max", type=float, default=1.0)
    train.add_argument("--rollout-horizon", type=int, default=3)
    train.add_argument(
        "--rollout-allocation-threshold",
        type=float,
        default=DEFAULT_ALLOCATION_THRESHOLD,
        help="allocation threshold used by closed-loop training rollouts",
    )
    train.add_argument(
        "--rollout-max-allocations-per-step",
        type=int,
        default=DEFAULT_MAX_ALLOCATIONS_PER_STEP,
        help="maximum new cognitive cells materialized by one training rollout step",
    )
    train.add_argument(
        "--semantic-pooling",
        choices=("mean-v1", "order-aware-v2"),
        default="order-aware-v2",
        help="TCT semantic pooling contract; order-aware-v2 preserves token order",
    )
    train.add_argument("--teacher-forcing-epochs", type=int, default=1)
    train.add_argument("--rollout-ramp-epochs", type=int, default=2)
    train.add_argument("--thought-capacity", type=int, default=128)
    train.add_argument("--max-display-tokens", type=int, default=1536)
    train.add_argument("--display-canvas-tokens", type=int, default=64)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--max-examples", type=int)
    train.add_argument("--max-validation-examples", type=int)
    train.add_argument("--no-shuffle", action="store_true")
    train.add_argument("--log-every-steps", type=int, default=100)
    train.add_argument("--checkpoint-every-steps", type=int, default=5000)
    train.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_full = subparsers.add_parser(
        "train-full",
        help="run Stage B full-parameter diffusion-backbone training with FSDP FULL_SHARD",
    )
    train_full.add_argument("--data", required=True)
    train_full.add_argument(
        "--validation-data",
        help=(
            "optional held-out trajectory JSONL; when omitted, validation examples are "
            "taken from --data entries with metadata.split=validation"
        ),
    )
    train_full.add_argument("--output-dir", required=True)
    train_full.add_argument("--model", default="GSAI-ML/iLLaDA-8B-Base")
    train_full.add_argument(
        "--device",
        choices=("auto", "cuda", "npu", "cpu"),
        default="auto",
        help="Stage B compute device; auto prefers CUDA, then Ascend NPU, then CPU",
    )
    train_full.add_argument("--resume")
    train_full.add_argument("--init-cid-checkpoint")
    train_full.add_argument("--dtype", choices=("bf16",), default="bf16")
    train_full.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="target total Stage B epochs, including epochs restored by --resume",
    )
    train_full.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="peak CID-module learning rate; backbone LR is scaled separately",
    )
    train_full.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.5,
        help="backbone LR multiplier; use a lower value for small-model retention when needed",
    )
    train_full.add_argument("--weight-decay", type=float, default=0.01)
    train_full.add_argument("--micro-batch-size", type=int, default=1)
    train_full.add_argument(
        "--target-global-batch-size",
        type=int,
        default=32,
        help="target transition batch; ignored when gradient accumulation is explicit",
    )
    train_full.add_argument("--gradient-accumulation-steps", type=int)
    train_full.add_argument(
        "--mlp-chunk-size",
        type=int,
        default=256,
        help="token chunk size for exact iLLaDA MLP evaluation",
    )
    train_full.add_argument(
        "--norm-chunk-size",
        type=int,
        default=256,
        help="token chunk size for exact iLLaDA RMSNorm evaluation",
    )
    train_full.add_argument("--warmup-ratio", type=float, default=0.03)
    train_full.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    train_full.add_argument("--max-grad-norm", type=float, default=1.0)
    train_full.add_argument("--timestep-min", type=float, default=0.05)
    train_full.add_argument("--timestep-max", type=float, default=1.0)
    train_full.add_argument("--rollout-horizon", type=int, default=3)
    train_full.add_argument(
        "--rollout-allocation-threshold",
        type=float,
        default=DEFAULT_ALLOCATION_THRESHOLD,
        help="allocation threshold used by closed-loop training rollouts",
    )
    train_full.add_argument(
        "--rollout-max-allocations-per-step",
        type=int,
        default=DEFAULT_MAX_ALLOCATIONS_PER_STEP,
        help="maximum new cognitive cells materialized by one training rollout step",
    )
    train_full.add_argument(
        "--semantic-pooling",
        choices=("mean-v1", "order-aware-v2"),
        default="order-aware-v2",
        help="must match the Stage A checkpoint semantic pooling contract",
    )
    train_full.add_argument("--teacher-forcing-epochs", type=int, default=0)
    train_full.add_argument("--rollout-ramp-epochs", type=int, default=0)
    train_full.add_argument("--thought-capacity", type=int, default=128)
    train_full.add_argument("--max-display-tokens", type=int, default=1536)
    train_full.add_argument("--display-canvas-tokens", type=int, default=64)
    train_full.add_argument("--seed", type=int, default=0)
    train_full.add_argument("--max-examples", type=int)
    train_full.add_argument("--max-validation-examples", type=int)
    train_full.add_argument("--no-shuffle", action="store_true")
    train_full.add_argument("--log-every-steps", type=int, default=100)
    train_full.add_argument("--checkpoint-every-steps", type=int, default=2500)
    train_full.add_argument(
        "--fsdp-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "offload FSDP parameter/gradient shards and optimizer state to host memory; "
            "intended for 24 GB GPUs and disabled by default"
        ),
    )
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
    elif args.command == "build-composed-training":
        _build_composed_training(args)
    elif args.command == "build-tool-restraint-training":
        _build_tool_restraint_training(args)
    elif args.command == "build-deep-tool-restraint-training":
        _build_deep_tool_restraint_training(args)
    elif args.command == "build-natural-interaction-training":
        _build_natural_interaction_training(args)
    elif args.command == "build-natural-public-training":
        _build_natural_public_training(args)
    elif args.command == "build-surface-diverse-training":
        _build_surface_diverse_training(args)
    elif args.command == "build-long-horizon-training":
        _build_long_horizon_training(args)
    elif args.command == "build-compositional-training":
        _build_compositional_training(args)

    elif args.command == "prepare-distillation":
        _prepare_distillation(args)
    elif args.command == "compile-distillation":
        _compile_distillation(args)
    elif args.command == "review-distillation":
        _review_distillation(args)
    elif args.command == "dataset-manifest":
        _dataset_manifest(args)
    elif args.command == "migrate-dataset-contract-v3":
        _migrate_dataset_contract_v3(args)
    elif args.command == "build-contract-v3-validation":
        _build_contract_v3_validation(args)
    elif args.command == "migrate-dataset-contract-v4":
        _migrate_dataset_contract_v4(args)
    elif args.command == "build-contract-v4-validation":
        _build_contract_v4_validation(args)
    elif args.command == "build-curated-v4-training":
        _build_curated_v4_training(args)
    elif args.command == "materialize-trajectory-mixture":
        _materialize_trajectory_mixture(args)
    elif args.command == "audit-training-data":
        _audit_training_data(args)
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
