from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

from cid.data import BindingTarget, DisplayTarget, GroundingTarget, ThoughtTarget, TrajectoryExample
from cid.grounding import Anchor, AnchorKind, CognitiveLink, LinkRelation, ObjectRef
from cid.state import CellLifecycle, CognitiveRole

torch = pytest.importorskip("torch")
nn = import_module("torch.nn")
cid_model = import_module("cid.model")

ILLaDACIDAdapter = cid_model.ILLaDACIDAdapter
ILLaDACIDConfig = cid_model.ILLaDACIDConfig
ILLaDATrajectoryTensorizer = cid_model.ILLaDATrajectoryTensorizer
cid_loss = cid_model.cid_loss


class TinyConfig:
    model_type = "illada"
    hidden_size = 16
    vocab_size = 64
    num_attention_heads = 4
    max_position_embeddings = 64


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(TinyConfig.hidden_size, TinyConfig.hidden_size, bias=False)

    def forward(self, *, inputs_embeds, attention_mask, return_dict):
        assert return_dict
        weights = attention_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        context = (inputs_embeds * weights).sum(dim=1, keepdim=True)
        context = context / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return SimpleNamespace(last_hidden_state=inputs_embeds + self.projection(context))


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.decoder = TinyDecoder()
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.decoder


class TinyTokenizer:
    pad_token_id = 1

    def __call__(self, text, *, add_special_tokens, return_tensors, padding=False):
        assert return_tensors == "pt"
        if isinstance(text, list):
            encoded = [self._ids(item, add_special_tokens=add_special_tokens) for item in text]
            width = max((len(item) for item in encoded), default=0)
            input_ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
            for row, item in enumerate(encoded):
                input_ids[row, : len(item)] = torch.tensor(item)
                attention_mask[row, : len(item)] = 1
            assert padding
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        assert not padding
        return {"input_ids": torch.tensor([self._ids(text, add_special_tokens=add_special_tokens)])}

    @staticmethod
    def _ids(text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [6 + (ord(char) % 50) for char in text]
        if add_special_tokens:
            ids = [2, *ids, 3]
        return ids or [4]


def make_trajectory() -> TrajectoryExample:
    latency_anchor = Anchor(
        anchor_id="a:latency",
        kind=AnchorKind.TEXT,
        value="latency",
        object_id="metric:latency",
    )
    return TrajectoryExample(
        example_id="train-1",
        prompt="Return the documented latency.",
        target_display="37",
        protected_facts={"instruction": "use the documented value"},
        source_descriptors=(
            {
                "name": "docs",
                "description": "read documentation",
                "arguments": (
                    {"name": "key", "kind": "string", "required": True},
                    {"name": "scope", "kind": "string", "required": True},
                ),
            },
        ),
        binding_targets=(
            BindingTarget(
                need_id="latency",
                source="docs",
                first_need_step=1,
                executable_step=2,
                arguments={"key": "latency_ms", "scope": "production"},
                argument_steps={"key": 1, "scope": 2},
                target_cells=(ObjectRef.cell("c1"),),
            ),
        ),
        thought_targets=(
            ThoughtTarget(
                step=0,
                slot=0,
                cell_id="c0",
                semantic_text="Understand the request.",
                roles={CognitiveRole.PLAN: 1.0},
                uncertainty=0.4,
                noise=0.5,
            ),
            ThoughtTarget(
                step=1,
                slot=0,
                cell_id="c0",
                semantic_text="The request requires documentation.",
                roles={CognitiveRole.PLAN: 0.7, CognitiveRole.INFORMATION_NEED: 0.3},
                uncertainty=0.3,
                noise=0.3,
                lifecycle=CellLifecycle.STABLE,
            ),
            ThoughtTarget(
                step=1,
                slot=1,
                cell_id="c1",
                semantic_text="Need the documented latency value.",
                roles={CognitiveRole.INFORMATION_NEED: 1.0},
                uncertainty=0.8,
                noise=1.0,
            ),
        ),
        display_targets=(DisplayTarget(step=1, text="37"),),
        grounding_targets=(
            GroundingTarget(
                step=1,
                cell_id="c1",
                anchors=(latency_anchor,),
                links=(
                    CognitiveLink(
                        relation=LinkRelation.REQUESTS,
                        target=ObjectRef.source("docs"),
                    ),
                ),
            ),
        ),
    )


def test_trajectory_tensorizer_runs_full_optimizer_step() -> None:
    adapter = ILLaDACIDAdapter(
        TinyBackbone(),
        ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=16),
        freeze_backbone=True,
    )
    tensorizer = ILLaDATrajectoryTensorizer(adapter, TinyTokenizer())
    sample = tensorizer.tensorize(make_trajectory(), source_step=0, timestep=1.0)

    assert sample.batch.thought_semantic.shape == (1, 4, TinyConfig.hidden_size)
    assert sample.batch.prompt_ids.shape[1] > 0
    assert sample.batch.display_ids.eq(5).all()
    assert sample.targets.allocation_targets[0, 1] == 1
    assert sample.targets.allocation_mask[0, 1]
    assert sample.targets.thought_mask[0, :2].all()
    assert sample.targets.source_targets[0, 1] == 0
    assert sample.targets.argument_presence_targets[0, 1, 0] == 1
    assert sample.targets.argument_presence_targets[0, 1, 1] == 0
    assert sample.targets.anchor_mask[0, 1, 0]
    assert sample.targets.link_mask[0, 1, 0]

    optimizer = torch.optim.AdamW(
        (parameter for parameter in adapter.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    before = adapter.output_heads.allocation_head.weight.detach().clone()
    output = adapter(sample.batch)
    losses = cid_loss(output, sample.targets)

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.argument_ground)
    assert torch.isfinite(losses.anchor_ground)
    assert torch.isfinite(losses.link_ground)
    optimizer.zero_grad()
    losses.total.backward()
    optimizer.step()

    assert not torch.equal(before, adapter.output_heads.allocation_head.weight)
    assert all(parameter.grad is None for parameter in adapter.backbone.parameters())
