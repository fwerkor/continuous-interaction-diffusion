from __future__ import annotations

import pytest

from cid.cli import _estimate_stage_a_full_activation_bytes


def test_stage_a_activation_estimate_scales_with_geometry() -> None:
    base = _estimate_stage_a_full_activation_bytes(
        batch_size=1,
        sequence_tokens=1024,
        layer_count=32,
        hidden_size=4096,
        intermediate_size=11008,
        element_size=2,
    )
    doubled_batch = _estimate_stage_a_full_activation_bytes(
        batch_size=2,
        sequence_tokens=1024,
        layer_count=32,
        hidden_size=4096,
        intermediate_size=11008,
        element_size=2,
    )
    assert doubled_batch == 2 * base
    assert base > 3 * 1024**3


def test_stage_a_activation_estimate_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _estimate_stage_a_full_activation_bytes(
            batch_size=0,
            sequence_tokens=1024,
            layer_count=32,
            hidden_size=4096,
            intermediate_size=11008,
            element_size=2,
        )
