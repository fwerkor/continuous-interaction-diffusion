from __future__ import annotations

from cid.model.loading import backbone_source_identity


def test_local_backbone_identity_is_stable_and_tracks_weight_changes(tmp_path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"llama"}\n', encoding="utf-8")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"a" * (192 * 1024))

    first = backbone_source_identity(str(tmp_path))
    assert first == backbone_source_identity(str(tmp_path))

    payload = bytearray(weight.read_bytes())
    payload[1024] ^= 1
    weight.write_bytes(payload)

    assert backbone_source_identity(str(tmp_path)) != first
