"""EOS burst, editing hysteresis and async evidence invalidation contracts."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from cid.model.diffusion import CIDDiffusionScheduler  # noqa: E402
from cid.model.display_revision import DisplayRevisionConfig, DisplayRevisionController  # noqa: E402

MASK, EOS, VOCAB = 62, 63, 64


def _canvas(*ids: int) -> torch.Tensor:
    result = torch.full((1, 64), MASK, dtype=torch.long)
    result[0, : len(ids)] = torch.tensor(ids)
    return result


def _logits(*ids: int) -> torch.Tensor:
    result = torch.full((1, 64, VOCAB), -12.0)
    for index, token in enumerate(ids):
        result[0, index, token] = 12.0
    return result


def _run(controller: DisplayRevisionController, x: torch.Tensor, y: torch.Tensor):
    return controller.refine_display(
        CIDDiffusionScheduler(MASK, EOS),
        x,
        y,
        reveal_fraction=0.125,
        revision_fraction=0.25,
        revision_margin=0.1,
        repetition_score_multiplier=1.0,
    )


def _context(step: int, revision: int = 0, percepts=()):
    return SimpleNamespace(
        step=step,
        facts=SimpleNamespace(revision=revision),
        sources=(),
        percepts=percepts,
    )


@pytest.mark.parametrize("length", [1, 2, 4, 8, 16, 17, 24, 32, 48, 62])
def test_confident_burst_requires_real_eos(length: int) -> None:
    model = DisplayRevisionController()
    expected = [2 + pos % 40 for pos in range(length)]
    x = _canvas(MASK, EOS)
    logits = _logits(*expected, EOS)
    for _ in range(64):
        x = _run(model, x, logits)
        if x[0, : length + 1].tolist() == expected + [EOS]:
            break
    assert x[0, : length + 1].tolist() == expected + [EOS]
    assert x[0, length + 1 :].eq(MASK).all()


def test_burst_rejects_low_confidence_middle() -> None:
    model = DisplayRevisionController()
    logits = _logits(4, 5, 6, EOS)
    logits[0, 2, :] = 0
    x = _run(model, _canvas(MASK, EOS), logits)
    assert x[0, :4].tolist() != [4, 5, 6, EOS]


def test_confident_output_survives_seven_wrong_proposals_but_stays_editable() -> None:
    model = DisplayRevisionController()
    x = _run(model, _canvas(MASK, EOS), _logits(4, 5, 6, EOS))
    assert x[0, :4].tolist() == [4, 5, 6, EOS]
    for _ in range(7):
        x = _run(model, x, _logits(4, 12, 6, EOS))
        assert x[0, :4].tolist() == [4, 5, 6, EOS]
    x = _run(model, x, _logits(4, 12, 6, EOS))
    assert x[0, :4].tolist() == [4, 12, 6, EOS]


def test_wrong_confident_eos_can_be_moved() -> None:
    model = DisplayRevisionController()
    x = _run(model, _canvas(MASK, EOS), _logits(4, 5, EOS))
    for _ in range(7):
        x = _run(model, x, _logits(4, 5, 6, EOS))
        assert x[0, 2] == EOS
    x = _run(model, x, _logits(4, 5, 6, EOS))
    assert x[0, :4].tolist() == [4, 5, 6, EOS]


def test_evidence_change_unlocks_without_discarding_display() -> None:
    model = DisplayRevisionController()
    model.observe_context(_context(0))
    x = _run(model, _canvas(MASK, EOS), _logits(4, 5, EOS))
    model.observe_context(_context(1))
    x = _run(model, x, _logits(4, 12, EOS))
    assert x[0, 1] == 5
    model.observe_context(_context(2, revision=1))
    x = _run(model, x, _logits(4, 12, EOS))
    assert x[0, 1] == 12
    model.observe_context(_context(3, revision=1, percepts=("new-data",)))
    assert model._trusted is None


def test_new_trajectory_resets_trust() -> None:
    model = DisplayRevisionController()
    model.observe_context(_context(0))
    _run(model, _canvas(MASK, EOS), _logits(4, 5, EOS))
    model.observe_context(_context(1))
    model.observe_context(_context(0))
    assert model._trusted is None
    assert _run(model, _canvas(MASK, EOS), _logits(10, 11, EOS))[0, :3].tolist() == [10, 11, EOS]


def test_repeated_tokens_are_allowed() -> None:
    model = DisplayRevisionController()
    assert _run(model, _canvas(MASK, EOS), _logits(5, 5, 5, EOS))[0, :4].tolist() == [5, 5, 5, EOS]


@pytest.mark.parametrize("bad", [0, -1, 1.2])
def test_rejects_bad_confidence(bad: float) -> None:
    with pytest.raises(ValueError):
        DisplayRevisionConfig(burst_probability=bad)


def test_configuration_does_not_mutate_caller_input() -> None:
    model = DisplayRevisionController()
    x = _canvas(MASK, EOS)
    copy = x.clone()
    _run(model, x, _logits(4, 5, EOS))
    assert torch.equal(x, copy)
