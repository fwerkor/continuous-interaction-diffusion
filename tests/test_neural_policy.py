from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

from cid.runtime import CIDRuntime, RuntimeConfig, SourceRegistry, StaticMappingSource
from cid.state import CognitiveField, DisplayCanvas

torch = pytest.importorskip("torch")
nn = import_module("torch.nn")
cid_model = import_module("cid.model")

CIDMaterializer = cid_model.CIDMaterializer
CIDMaterializerConfig = cid_model.CIDMaterializerConfig
ClosedWorldMaterializationCatalog = cid_model.ClosedWorldMaterializationCatalog
ArgumentCandidate = cid_model.ArgumentCandidate
ILLaDACIDAdapter = cid_model.ILLaDACIDAdapter
ILLaDAContextTensorizer = cid_model.ILLaDAContextTensorizer
ILLaDANeuralPolicy = cid_model.ILLaDANeuralPolicy
ILLaDANeuralPolicyConfig = cid_model.ILLaDANeuralPolicyConfig


class TinyConfig:
    model_type = "illada"
    hidden_size = 16
    vocab_size = 32
    num_attention_heads = 4
    max_position_embeddings = 32


class IdentityDecoder(nn.Module):
    def forward(self, *, inputs_embeds, attention_mask, return_dict):
        del attention_mask
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
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, text, *, add_special_tokens, return_tensors, padding=False):
        assert return_tensors == "pt"
        if isinstance(text, list):
            encoded = [self._ids(item) for item in text]
            width = max((len(item) for item in encoded), default=0)
            input_ids = torch.ones((len(encoded), width), dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
            for row, item in enumerate(encoded):
                if item:
                    input_ids[row, : len(item)] = torch.tensor(item)
                    attention_mask[row, : len(item)] = 1
            assert padding
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        assert add_special_tokens
        assert not padding
        self.prompts.append(text)
        return {"input_ids": torch.tensor([[3, *self._ids(text), 4]])}

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [6 + (ord(char) % 20) for char in text[:8]]


async def test_neural_policy_runs_tiny_illada_inside_async_runtime() -> None:
    adapter = ILLaDACIDAdapter(TinyBackbone(), freeze_backbone=True)
    tokenizer = TinyTokenizer()
    tensorizer = ILLaDAContextTensorizer(adapter, tokenizer)
    materializer = CIDMaterializer(
        CIDMaterializerConfig(
            allocation_threshold=1.0,
            need_threshold=1.0,
            anchor_presence_threshold=1.0,
            link_presence_threshold=1.0,
        )
    )
    policy = ILLaDANeuralPolicy(
        adapter,
        tensorizer,
        materializer=materializer,
        config=ILLaDANeuralPolicyConfig(denoising_steps=1),
    )
    thought = CognitiveField.empty(capacity=2, width=TinyConfig.hidden_size)
    thought, _ = thought.allocate(semantic=(0.0,) * TinyConfig.hidden_size)
    runtime = CIDRuntime(SourceRegistry(), RuntimeConfig(max_steps=3))

    result = await runtime.run(
        policy,
        thought=thought,
        display=DisplayCanvas.masked(length=3, mask_token_id=5),
        prompt="Which value should I return?",
    )

    assert result.converged
    assert result.steps == 1
    assert result.display.token_ids == (7, 7, 7)
    assert tokenizer.prompts == ["Which value should I return?"]


async def test_neural_policy_materializes_executable_need_and_reads_source() -> None:
    adapter = ILLaDACIDAdapter(TinyBackbone(), freeze_backbone=True)
    with torch.no_grad():
        adapter.output_heads.allocation_head.weight.zero_()
        adapter.output_heads.allocation_head.bias.fill_(-10.0)
        adapter.output_heads.need_head.weight.zero_()
        adapter.output_heads.need_head.bias.fill_(10.0)
        adapter.output_heads.argument_presence_head.weight.zero_()
        adapter.output_heads.argument_presence_head.bias.fill_(-10.0)
        adapter.output_heads.argument_presence_head.bias[0] = 10.0
        adapter.output_heads.argument_query.weight.zero_()
        adapter.output_heads.anchor_presence_head.weight.zero_()
        adapter.output_heads.anchor_presence_head.bias.fill_(-10.0)
        adapter.output_heads.link_presence_head.weight.zero_()
        adapter.output_heads.link_presence_head.bias.fill_(-10.0)
        adapter.output_heads.lifecycle_head.weight.zero_()
        adapter.output_heads.lifecycle_head.bias.zero_()
        adapter.output_heads.lifecycle_head.bias[0] = 10.0
        adapter.output_heads.revision_head.weight.zero_()
        adapter.output_heads.revision_head.bias.zero_()
        adapter.output_heads.revision_head.bias[0] = 10.0
        adapter.output_heads.refresh_head.weight.zero_()
        adapter.output_heads.refresh_head.bias.zero_()
        adapter.output_heads.refresh_head.bias[0] = 10.0

    tokenizer = TinyTokenizer()
    tensorizer = ILLaDAContextTensorizer(adapter, tokenizer)
    catalog = ClosedWorldMaterializationCatalog(
        arguments=(
            ArgumentCandidate(
                source="docs",
                name="key",
                value="latency_ms",
                embedding=torch.ones(TinyConfig.hidden_size),
            ),
        )
    )
    policy = ILLaDANeuralPolicy(
        adapter,
        tensorizer,
        catalog=catalog,
        materializer=CIDMaterializer(
            CIDMaterializerConfig(
                allocation_threshold=1.0,
                need_threshold=0.8,
                argument_presence_threshold=0.8,
                anchor_presence_threshold=1.0,
                link_presence_threshold=1.0,
                retrieval_similarity_threshold=-1.0,
            )
        ),
        config=ILLaDANeuralPolicyConfig(denoising_steps=1),
    )
    sources = SourceRegistry()
    sources.register(StaticMappingSource(name="docs", values={"latency_ms": 37}))
    thought = CognitiveField.empty(capacity=2, width=TinyConfig.hidden_size)
    thought, _ = thought.allocate(semantic=(0.0,) * TinyConfig.hidden_size)

    result = await CIDRuntime(sources, RuntimeConfig(max_steps=4)).run(
        policy,
        thought=thought,
        display=DisplayCanvas.masked(length=2, mask_token_id=5),
        prompt="Read the latency.",
    )

    assert result.converged
    assert result.bindings
    assert result.bindings[0].arguments == {"key": "latency_ms"}
    assert result.bindings[0].external_refreshes == 1
    assert result.bindings[0].observation is not None
    assert result.bindings[0].observation.value == 37


def test_context_tensorizer_requires_illada_mask_token() -> None:
    adapter = ILLaDACIDAdapter(TinyBackbone())
    tensorizer = ILLaDAContextTensorizer(adapter, TinyTokenizer())
    thought = CognitiveField.empty(capacity=2, width=TinyConfig.hidden_size)

    context = SimpleNamespace(
        thought=thought,
        display=DisplayCanvas.masked(length=2, mask_token_id=6),
        facts=SimpleNamespace(items={}),
        percepts=(),
        sources=(),
    )

    with pytest.raises(ValueError, match="mask token id 5"):
        tensorizer(context)
