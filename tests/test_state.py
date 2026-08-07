import time

import pytest

from cid.state import FactItem, FactStore


def test_fact_snapshot_is_read_only() -> None:
    store = FactStore()
    store.publish(
        FactItem(
            key="exact",
            value="37 ms",
            source_type="docs",
            timestamp=time.monotonic(),
        )
    )
    snapshot = store.snapshot()

    with pytest.raises(TypeError):
        snapshot.items["exact"] = snapshot.items["exact"]  # type: ignore[index]

    assert store.snapshot().items["exact"].value == "37 ms"


def test_fact_snapshot_cannot_mutate_nested_runtime_value() -> None:
    store = FactStore()
    store.publish(
        FactItem(
            key="structured",
            value={"numbers": [37]},
            source_type="docs",
            timestamp=time.monotonic(),
        )
    )

    snapshot = store.snapshot()
    snapshot.items["structured"].value["numbers"].append(99)

    assert store.snapshot().items["structured"].value == {"numbers": [37]}
