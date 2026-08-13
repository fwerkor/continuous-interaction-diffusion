from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REFERENCE_KEYS = frozenset(
    {
        "causal_jobs",
        "input_pool",
        "manifest",
        "path",
        "reference_manifest",
        "seed_trajectory_manifest",
        "semantic_mixture",
        "trajectory_manifest",
    }
)
RELEASE_PREFIXES = ("data/", "manifests/", "metadata/")
SURFACE_SUMMARY_KEYS = frozenset(
    {
        "largest_normalized_prompt_group",
        "largest_normalized_semantic_text_group",
        "normalized_prompt_signatures",
        "normalized_semantic_text_signatures",
        "semantic_text_fallback_plans",
    }
)


@dataclass(frozen=True, slots=True)
class ManifestReferenceIssue:
    manifest_path: str
    field_path: str
    target: str


@dataclass(frozen=True, slots=True)
class ManifestSummaryIssue:
    component: str
    key: str
    declared: Any
    actual: Any


def find_missing_manifest_references(
    manifests: Mapping[str, Any],
    available_paths: Iterable[str],
) -> tuple[ManifestReferenceIssue, ...]:
    available = set(available_paths)
    issues: list[ManifestReferenceIssue] = []
    for manifest_path, payload in sorted(manifests.items()):
        for field_path, target in iter_manifest_references(payload):
            if target not in available:
                issues.append(
                    ManifestReferenceIssue(
                        manifest_path=manifest_path,
                        field_path=field_path,
                        target=target,
                    )
                )
    return tuple(issues)


def iter_manifest_references(payload: Any) -> tuple[tuple[str, str], ...]:
    references: list[tuple[str, str]] = []

    def visit(value: Any, field_path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{field_path}.{key_text}" if field_path else key_text
                if (
                    key_text in REFERENCE_KEYS
                    and isinstance(child, str)
                    and child.startswith(RELEASE_PREFIXES)
                ):
                    references.append((child_path, child))
                visit(child, child_path)
        elif isinstance(value, list | tuple):
            for index, child in enumerate(value):
                visit(child, f"{field_path}[{index}]")

    visit(payload, "")
    return tuple(references)


def load_release_json_manifests(
    root: str | Path,
    available_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    selected = (
        set(available_paths)
        if available_paths is not None
        else {
            str(path.relative_to(root_path)).replace("\\", "/")
            for path in root_path.rglob("*.json")
        }
    )
    manifests: dict[str, Any] = {}
    for relative in sorted(selected):
        if not relative.endswith(".json"):
            continue
        path = root_path / relative
        if not path.is_file():
            continue
        try:
            manifests[relative] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid release JSON {relative}: {exc}") from exc
    return manifests


def assert_release_references_resolve(
    root: str | Path,
    available_paths: Iterable[str],
) -> None:
    available = tuple(available_paths)
    manifests = load_release_json_manifests(root, available)
    issues = find_missing_manifest_references(manifests, available)
    if not issues:
        return
    detail = "; ".join(
        f"{issue.manifest_path}:{issue.field_path}->{issue.target}" for issue in issues[:20]
    )
    suffix = "" if len(issues) <= 20 else f"; ... {len(issues) - 20} more"
    raise ValueError(
        f"release contains {len(issues)} missing manifest references: {detail}{suffix}"
    )


def find_surface_summary_mismatches(
    root: str | Path,
    semantic_mixture: str | Path,
) -> tuple[ManifestSummaryIssue, ...]:
    """Check cached surface-diversity summary fields against their reference manifests."""

    root_path = Path(root)
    mixture_path = root_path / semantic_mixture
    mixture = json.loads(mixture_path.read_text(encoding="utf-8"))
    issues: list[ManifestSummaryIssue] = []
    for component, summary in mixture.get("surface_diverse_replacements", {}).items():
        reference = summary.get("reference_manifest")
        if not isinstance(reference, str):
            continue
        manifest = json.loads((root_path / reference).read_text(encoding="utf-8"))
        for key in sorted(SURFACE_SUMMARY_KEYS & summary.keys() & manifest.keys()):
            if summary[key] != manifest[key]:
                issues.append(
                    ManifestSummaryIssue(
                        component=str(component),
                        key=key,
                        declared=summary[key],
                        actual=manifest[key],
                    )
                )
    return tuple(issues)
