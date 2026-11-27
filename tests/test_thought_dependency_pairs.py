from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from cid.data import DISPLAY_UNKNOWN_MARKER, TrajectoryExample
from cid.model.encoding import ILLaDATextEncoder
from cid.model.illada import ILLaDACIDConfig
from cid.model.lfm import LFMCIDAdapter
from cid.model.losses import cid_loss
from cid.model.training import ILLaDATrajectoryTensorizer
from scripts.build_thought_dependency_pairs import example
from test_lfm_adapter import TinyLFMBackbone
from test_stage_ab_language_mix import ToyTokenizer


class NativeTestTokenizer(ToyTokenizer):
    chat_template = "native-test"
    eos_token_id = 7
    all_special_ids = (0, 7, 16)

    def __len__(self):
        return 64

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        return "User: " + messages[0]["content"] + "\nAssistant: "


def test_counterfactual_pairs_require_thought_to_distinguish_display() -> None:
    left = TrajectoryExample.from_dict(example(1, 204, 0, 147, split="train"))
    right = TrajectoryExample.from_dict(example(1, 204, 1, 213, split="train"))
    assert left.prompt == right.prompt
    assert left.target_display != right.target_display
    assert left.thought_targets[1].semantic_text != right.thought_targets[1].semantic_text
    assert left.display_targets[0].text == DISPLAY_UNKNOWN_MARKER
    for source in (left, right):
        assert source.metadata["native_full_display_by_step"]["1"] == source.target_display

    adapter = LFMCIDAdapter(
        TinyLFMBackbone(),
        ILLaDACIDConfig(
            max_thought_slots=8,
            max_display_tokens=96,
            display_canvas_tokens=16,
            interaction_architecture="identity-preserving",
            backbone_protocol="lfm-native",
            semantic_head_version=2,
        ),
        freeze_backbone=True,
    )
    tokenizer = NativeTestTokenizer()
    tensorizer = ILLaDATrajectoryTensorizer(
        adapter, tokenizer, text_encoder=ILLaDATextEncoder(adapter, tokenizer)
    )
    for trajectory in (left, right):
        bootstrap = tensorizer.tensorize(trajectory, -1, timestep=1.0)
        # An unresolved Display must not be trained to hallucinate a result.
        assert not bootstrap.targets.display_ids.ne(-100).any()
        conditioned = tensorizer.tensorize(trajectory, 0, timestep=0.5)
        assert conditioned.batch.slot_occupancy[0].sum().item() == 2.0
        assert conditioned.targets.display_ids.ne(-100).any()
        assert conditioned.targets.thought_mask.any()
        out = adapter(conditioned.batch)
        loss = cid_loss(out, conditioned.targets)
        assert torch.isfinite(loss.total)
