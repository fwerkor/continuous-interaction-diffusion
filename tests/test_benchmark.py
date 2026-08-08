from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

from cid.data import BindingTarget, ExternalEvent, ThoughtTarget, TrajectoryExample
from cid.grounding import ObjectRef
from cid.state import CellLifecycle, CognitiveRole

torch = pytest.importorskip("torch")
nn = import_module("torch.nn")
benchmark_module = import_module("cid.model.benchmark")
cid_model = import_module("cid.model")

ILLaDACIDAdapter = cid_model.ILLaDACIDAdapter
run_neural_benchmark_case = benchmark_module.run_neural_benchmark_case
build_materialization_catalog = benchmark_module.build_materialization_catalog
teacher_seed_thought = benchmark_module.teacher_seed_thought


class TinyConfig:
    model_type = "illada"
    hidden_size = 16
    vocab_size = 32
    num_attention_heads = 4
    max_position_embeddings = 32


class IdentityDecoder(nn.Module):
    def forward(self, *, inputs_embeds, attention_mask, position_ids, return_dict):
        del attention_mask, position_ids
        assert return_dict
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class FixedHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, token_id: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        with torch.no_grad():
            self.bias[token_id] = 10.0

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden, self.weight, self.bias)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.decoder = IdentityDecoder()
        self.lm_head = FixedHead(self.config.hidden_size, self.config.vocab_size, token_id=7)

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
            encoded = [self._ids(item) for item in text]
            width = max((len(item) for item in encoded), default=0)
            input_ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
            for row, item in enumerate(encoded):
                if item:
                    input_ids[row, : len(item)] = torch.tensor(item)
                    attention_mask[row, : len(item)] = 1
            assert padding
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        assert not padding
        ids = self._ids(text)
        if add_special_tokens:
            ids = [2, *ids, 3]
        return {"input_ids": torch.tensor([ids])}

    @staticmethod
    def _ids(text: str) -> list[int]:
        if text == "x":
            return [7]
        return [6 + (ord(char) % 20) for char in text[:8]] or [4]

    @staticmethod
    def decode(ids, *, skip_special_tokens):
        del skip_special_tokens
        return " ".join(str(item) for item in ids)


def make_example() -> TrajectoryExample:
    return TrajectoryExample(
        example_id="bench-1",
        prompt="Read the value.",
        target_display="x",
        source_descriptors=(
            {
                "name": "docs",
                "description": "read docs",
                "arguments": ({"name": "key", "kind": "string", "required": True},),
            },
        ),
        events=(
            ExternalEvent(
                source="docs",
                value=37,
                arrival_step=1,
                version="v1",
                arguments={"key": "latency"},
            ),
        ),
        binding_targets=(
            BindingTarget(
                need_id="latency",
                source="docs",
                first_need_step=0,
                executable_step=0,
                arguments={"key": "latency"},
                target_cells=(ObjectRef.cell("c0"),),
            ),
        ),
        thought_targets=(
            ThoughtTarget(
                step=0,
                slot=0,
                cell_id="c0",
                semantic_text="Need documentation.",
                roles={CognitiveRole.INFORMATION_NEED: 1.0},
                lifecycle=CellLifecycle.ACTIVE,
            ),
            ThoughtTarget(
                step=1,
                slot=0,
                cell_id="c0",
                semantic_text="Observed documentation.",
                roles={CognitiveRole.CONCLUSION: 1.0},
                lifecycle=CellLifecycle.STABLE,
            ),
        ),
    )


def configured_adapter() -> ILLaDACIDAdapter:
    adapter = ILLaDACIDAdapter(
        TinyBackbone(),
        cid_model.ILLaDACIDConfig(max_thought_slots=4, max_display_tokens=8),
        freeze_backbone=True,
    )
    with torch.no_grad():
        adapter.output_heads.need_head.weight.zero_()
        adapter.output_heads.need_head.bias.fill_(10.0)
        adapter.output_heads.argument_presence_head.weight.zero_()
        adapter.output_heads.argument_presence_head.bias.fill_(-10.0)
        adapter.output_heads.argument_presence_head.bias[0] = 10.0
        adapter.output_heads.argument_query.weight.zero_()
        adapter.output_heads.convergence_head.weight.zero_()
        adapter.output_heads.convergence_head.bias.fill_(10.0)
        adapter.output_heads.allocation_head.weight.zero_()
        adapter.output_heads.allocation_head.bias.fill_(-10.0)
        adapter.output_heads.anchor_presence_head.bias.fill_(-10.0)
        adapter.output_heads.link_presence_head.bias.fill_(-10.0)
    return adapter


def test_benchmark_catalog_and_teacher_seed_are_deterministic() -> None:
    adapter = configured_adapter()
    tokenizer = TinyTokenizer()
    encoder = import_module("cid.model.encoding").ILLaDATextEncoder(adapter, tokenizer)
    example = make_example()

    catalog = build_materialization_catalog(example, encoder)
    thought = teacher_seed_thought(example, adapter, encoder)

    assert len(catalog.arguments) == 1
    assert catalog.arguments[0].value == "latency"
    assert thought.live_cell_ids == ("c0",)
    assert thought.slot_of("c0") == 0


async def test_neural_benchmark_case_runs_replay_and_scores_display() -> None:
    adapter = configured_adapter()
    example = make_example()
    display_only = TrajectoryExample(
        example_id=example.example_id,
        prompt=example.prompt,
        target_display=example.target_display,
        thought_targets=example.thought_targets,
    )
    result = await run_neural_benchmark_case(
        adapter,
        TinyTokenizer(),
        display_only,
        seed_teacher_state=True,
        denoising_steps=1,
        max_steps=4,
    )

    assert result.example_id == "bench-1"
    assert result.final_display_ids == (7,)
    assert result.evaluation.exact_display is True
    assert result.evaluation.expected_observations == 0
    assert result.evaluation.observation_coverage == 1.0
