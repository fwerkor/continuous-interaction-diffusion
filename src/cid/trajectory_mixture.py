from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cid.data import adjacent_transition_source_steps, training_transition_source_steps


@dataclass(frozen=True, slots=True)
class TrajectoryMixtureComponent:
    name: str
    path: str
    manifest: str
    sha256: str
    examples: int
    transitions: int
    bootstrap_transitions: int | None = None
    training_transitions: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrajectoryMixtureComponent:
        return cls(
            name=str(raw["name"]),
            path=str(raw["path"]),
            manifest=str(raw["manifest"]),
            sha256=str(raw["sha256"]),
            examples=int(raw["examples"]),
            transitions=int(raw["transitions"]),
            bootstrap_transitions=(
                int(raw["bootstrap_transitions"])
                if "bootstrap_transitions" in raw
                else None
            ),
            training_transitions=(
                int(raw["training_transitions"])
                if "training_transitions" in raw
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryMixtureSpec:
    name: str
    version: int
    components: tuple[TrajectoryMixtureComponent, ...]

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryMixtureSpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        components = tuple(
            TrajectoryMixtureComponent.from_dict(item) for item in raw.get("components", ())
        )
        if not components:
            raise ValueError("trajectory mixture requires at least one component")
        names = [item.name for item in components]
        if len(names) != len(set(names)):
            raise ValueError("trajectory mixture component names must be unique")
        return cls(name=str(raw["name"]), version=int(raw["version"]), components=components)


def materialize_trajectory_mixture(
    spec_path: str | Path,
    output_path: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    spec_file = Path(spec_path)
    spec = TrajectoryMixtureSpec.load(spec_file)
    spec_parent = spec_file.resolve().parent
    root = (
        spec_parent.parent if spec_parent.name in {"data", "manifests", "metadata"} else spec_parent
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    output_sha = hashlib.sha256()
    output_bytes = 0
    output_examples = 0
    expected_transitions = 0
    expected_bootstrap_transitions = 0
    expected_training_transitions = 0
    trainable_examples = 0
    zero_training_transition_examples = 0
    tag_counts: Counter[str] = Counter()
    sources: set[str] = set()
    thought_capacity_required = 0
    max_trajectory_steps = 0
    component_records: list[dict[str, Any]] = []

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)
        try:
            for component in spec.components:
                source = _resolve(root, component.path)
                component_manifest_path = _resolve(root, component.manifest)
                component_manifest = json.loads(component_manifest_path.read_text(encoding="utf-8"))
                if component_manifest.get("sha256") != component.sha256:
                    raise ValueError(
                        f"component {component.name} manifest SHA does not match mixture spec"
                    )
                if int(component_manifest.get("examples", -1)) != component.examples:
                    raise ValueError(
                        f"component {component.name} manifest example count "
                        "does not match mixture spec"
                    )
                if int(component_manifest.get("transitions", -1)) != component.transitions:
                    raise ValueError(
                        f"component {component.name} manifest transition count "
                        "does not match mixture spec"
                    )
                source_sha = hashlib.sha256()
                line_count = 0
                component_transitions = 0
                component_bootstrap_transitions = 0
                component_training_transitions = 0
                component_trainable_examples = 0
                component_zero_training_transition_examples = 0
                with source.open("rb") as handle:
                    for raw_line in handle:
                        source_sha.update(raw_line)
                        stripped = raw_line.strip()
                        if not stripped:
                            temp.write(raw_line)
                            output_sha.update(raw_line)
                            output_bytes += len(raw_line)
                            continue
                        try:
                            record = json.loads(stripped)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"invalid trajectory JSON in {component.name} "
                                f"line {line_count + 1}: {exc}"
                            ) from exc
                        example_id = str(record.get("example_id", ""))
                        if not example_id:
                            raise ValueError(
                                f"component {component.name} contains an empty example_id"
                            )
                        if example_id in seen_ids:
                            raise ValueError(
                                f"duplicate trajectory example_id across mixture: {example_id}"
                            )
                        seen_ids.add(example_id)
                        line_count += 1

                        metadata = record.get("metadata", {})
                        family = metadata.get("family")
                        distillation = metadata.get("distillation")
                        if family:
                            tag_counts[f"family:{family}"] += 1
                        elif distillation:
                            tag_counts[f"distillation:{distillation}"] += 1
                        else:
                            tag_counts["unlabeled"] += 1
                        sources.update(
                            str(descriptor.get("name", ""))
                            for descriptor in record.get("source_descriptors", ())
                            if str(descriptor.get("name", ""))
                        )
                        thought_targets = record.get("thought_targets", ())
                        steps = {int(target["step"]) for target in thought_targets}
                        adjacent_sources = adjacent_transition_source_steps(steps)
                        training_sources = training_transition_source_steps(steps)
                        component_transitions += len(adjacent_sources)
                        component_bootstrap_transitions += int(-1 in training_sources)
                        component_training_transitions += len(training_sources)
                        if training_sources:
                            component_trainable_examples += 1
                        else:
                            component_zero_training_transition_examples += 1
                        thought_capacity_required = max(
                            thought_capacity_required,
                            max((int(target["slot"]) + 1 for target in thought_targets), default=0),
                        )
                        max_trajectory_steps = max(max_trajectory_steps, len(steps))

                        temp.write(raw_line)
                        output_sha.update(raw_line)
                        output_bytes += len(raw_line)
                if source_sha.hexdigest() != component.sha256:
                    raise ValueError(
                        f"component {component.name} file SHA does not match mixture spec"
                    )
                if line_count != component.examples:
                    raise ValueError(
                        f"component {component.name} example count {line_count} "
                        f"!= {component.examples}"
                    )
                if component_transitions != component.transitions:
                    raise ValueError(
                        f"component {component.name} transition count {component_transitions} "
                        f"!= {component.transitions}"
                    )
                if (
                    component.bootstrap_transitions is not None
                    and component_bootstrap_transitions != component.bootstrap_transitions
                ):
                    raise ValueError(
                        f"component {component.name} bootstrap transition count "
                        f"{component_bootstrap_transitions} != {component.bootstrap_transitions}"
                    )
                if (
                    component.training_transitions is not None
                    and component_training_transitions != component.training_transitions
                ):
                    raise ValueError(
                        f"component {component.name} training transition count "
                        f"{component_training_transitions} != {component.training_transitions}"
                    )
                output_examples += line_count
                expected_transitions += component_transitions
                expected_bootstrap_transitions += component_bootstrap_transitions
                expected_training_transitions += component_training_transitions
                trainable_examples += component_trainable_examples
                zero_training_transition_examples += component_zero_training_transition_examples
                component_records.append(
                    {
                        "name": component.name,
                        "path": component.path,
                        "sha256": component.sha256,
                        "examples": component.examples,
                        "transitions": component.transitions,
                        "bootstrap_transitions": component_bootstrap_transitions,
                        "training_transitions": component_training_transitions,
                        "trainable_examples": component_trainable_examples,
                        "zero_training_transition_examples": (
                            component_zero_training_transition_examples
                        ),
                    }
                )
            temp.flush()
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(destination)

    manifest = {
        "format_version": 1,
        "schema": "cid.TrajectoryExample.v1",
        "mixture": spec.name,
        "mixture_version": spec.version,
        "mixture_spec_sha256": hashlib.sha256(spec_file.read_bytes()).hexdigest(),
        "merge_mode": "component-order-concatenation; trainer shuffles rollout windows per epoch",
        "sha256": output_sha.hexdigest(),
        "bytes": output_bytes,
        "examples": output_examples,
        "transitions": expected_transitions,
        "bootstrap_transitions": expected_bootstrap_transitions,
        "training_transitions": expected_training_transitions,
        "trainable_examples": trainable_examples,
        "zero_training_transition_examples": zero_training_transition_examples,
        "tag_counts": dict(sorted(tag_counts.items())),
        "sources": sorted(sources),
        "thought_capacity_required": thought_capacity_required,
        "max_trajectory_steps": max_trajectory_steps,
        "components": component_records,
    }
    manifest_path = Path(manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    return manifest


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
