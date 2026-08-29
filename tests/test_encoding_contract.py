import pytest

pytest.importorskip("torch")

from cid.contracts import ArgumentDescriptor, SourceDescriptor
from cid.grounding import ObjectRef
from cid.model.encoding import (
    canonical_fact_text,
    canonical_percept_text,
    canonical_source_text,
)
from cid.state import FactItem


def test_source_text_is_identical_for_dataset_and_runtime_descriptors() -> None:
    raw = {
        "name": "docs",
        "description": "read documentation",
        "arguments": (
            {"name": "key", "kind": "string", "required": True},
            {"name": "scope", "kind": "string", "required": False},
        ),
        "cacheable": False,
        "dynamic": True,
        "streamable": True,
        "versioned": True,
        "accepts_partial_arguments": True,
    }
    runtime = SourceDescriptor(
        name="docs",
        description="read documentation",
        arguments=(
            ArgumentDescriptor(name="key", kind="string", required=True),
            ArgumentDescriptor(name="scope", kind="string", required=False),
        ),
        cacheable=False,
        dynamic=True,
        streamable=True,
        versioned=True,
        accepts_partial_arguments=True,
    )

    assert canonical_source_text(raw) == canonical_source_text(runtime)


def test_fact_text_is_identical_for_dataset_and_runtime_facts() -> None:
    runtime = FactItem(
        key="instruction",
        value={"mode": "exact"},
        source_type="dataset",
        timestamp=123.0,
    )

    assert canonical_fact_text(runtime) == canonical_fact_text(
        key="instruction",
        value={"mode": "exact"},
        source_type="dataset",
    )


def test_percept_text_depends_on_semantics_not_runtime_binding_serial() -> None:
    target_cells = (ObjectRef.cell("c2"),)
    target_display = (ObjectRef.display_span(1, 3),)

    first = canonical_percept_text(
        source="docs",
        value={"latency": 37},
        version="v2",
        target_cells=target_cells,
        target_display=target_display,
    )
    second = canonical_percept_text(
        source="docs",
        value={"latency": 37},
        version="v2",
        target_cells=target_cells,
        target_display=target_display,
    )

    assert first == second
    assert "binding" not in first
