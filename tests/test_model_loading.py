from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from cid.model.loading import (
    apply_neural_contract_compatibility,
    backbone_identity_matches,
    backbone_source_identity,
    tokenizer_source_identity,
)


def test_local_backbone_identity_is_stable_and_tracks_middle_weight_changes(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "model"
    source.mkdir()
    cache = tmp_path / "cache"
    monkeypatch.setenv("CID_SOURCE_IDENTITY_CACHE_DIR", str(cache))
    (source / "config.json").write_text('{"model_type":"llama"}\n', encoding="utf-8")
    weight = source / "model.safetensors"
    weight.write_bytes(b"a" * (512 * 1024))

    first = backbone_source_identity(str(source))
    assert first.startswith("local-sha256:")
    assert first == backbone_source_identity(str(source))

    payload = bytearray(weight.read_bytes())
    payload[len(payload) // 2] ^= 1
    weight.write_bytes(payload)

    assert backbone_source_identity(str(source)) != first


def test_local_backbone_identity_recovers_from_non_object_cache(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "model"
    source.mkdir()
    cache = tmp_path / "cache"
    monkeypatch.setenv("CID_SOURCE_IDENTITY_CACHE_DIR", str(cache))
    (source / "config.json").write_text('{"model_type":"llama"}\n', encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    expected = backbone_source_identity(str(source))
    cache_file = next(cache.glob("*.json"))
    cache_file.write_text("[]\n", encoding="utf-8")

    assert backbone_source_identity(str(source)) == expected


def test_hf_backbone_identity_resolves_requested_revision_to_commit(monkeypatch) -> None:
    import huggingface_hub

    calls: list[tuple[str, str | None]] = []

    def model_info(self, repo_id, *, revision=None, **kwargs):
        del self, kwargs
        calls.append((repo_id, revision))
        return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)

    identity = backbone_source_identity("example/model", revision="experiment-branch")

    assert calls == [("example/model", "experiment-branch")]
    assert identity == f"hf:example/model@{'a' * 40}"


def test_backbone_identity_accepts_legacy_v5_identity() -> None:
    adapter = SimpleNamespace(
        _cid_backbone_identity="local-sha256:new",
        _cid_legacy_backbone_identity="local-edge-sha256:old",
    )

    assert backbone_identity_matches(adapter, "local-sha256:new")
    assert backbone_identity_matches(adapter, "local-edge-sha256:old")
    assert not backbone_identity_matches(adapter, "local-edge-sha256:other")


def test_unpinned_legacy_hf_identity_is_not_accepted(monkeypatch) -> None:
    adapter = SimpleNamespace(
        _cid_backbone_identity="hf:example/model@" + "a" * 40,
        _cid_legacy_backbone_identity=None,
    )

    assert not backbone_identity_matches(adapter, "hf:example/model@unversioned")


class _SavedTokenizer:
    def __init__(self, token: str, source: str) -> None:
        self.token = token
        self.source = source

    def save_pretrained(self, root) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer.json").write_text(
            json.dumps(
                {
                    "model": {"type": "WordLevel", "vocab": {self.token: 0}},
                    "name_or_path": self.source,
                }
            ),
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "name_or_path": self.source,
                    "vocab_file": f"{self.source}/vocab.json",
                }
            ),
            encoding="utf-8",
        )


def test_tokenizer_identity_tracks_content_not_source_path() -> None:
    first = tokenizer_source_identity(_SavedTokenizer("alpha", "/one/location"))
    same = tokenizer_source_identity(_SavedTokenizer("alpha", "/another/location"))
    different = tokenizer_source_identity(_SavedTokenizer("beta", "/one/location"))

    assert first == same
    assert first != different


def test_legacy_v4_release_compatibility_restores_original_semantics() -> None:
    adapter = SimpleNamespace(
        external_fusion=SimpleNamespace(percept_residual=True),
    )
    state = {"format_version": 1}

    compatible = apply_neural_contract_compatibility(
        adapter,
        state,
        neural_contract_version=4,
    )

    assert not adapter.external_fusion.percept_residual
    assert compatible["semantic_noise_scale"] == pytest.approx(1.0)
    assert "semantic_noise_scale" not in state


def test_v5_release_requires_explicit_semantic_noise_scale() -> None:
    adapter = SimpleNamespace(
        external_fusion=SimpleNamespace(percept_residual=True),
    )

    with pytest.raises(ValueError, match="semantic_noise_scale"):
        apply_neural_contract_compatibility(
            adapter,
            {"format_version": 1},
            neural_contract_version=5,
        )
