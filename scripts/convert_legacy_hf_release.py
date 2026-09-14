#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from cid.model import build_unified_cid_config, unified_state_from_legacy

REMOTE_CONFIGURATION = 'from cid.model.huggingface import CIDConfig\n\n__all__ = ["CIDConfig"]\n'
REMOTE_MODELING = 'from cid.model.huggingface import CIDModel\n\n__all__ = ["CIDModel"]\n'


def _source_directory(source: str, revision: str | None) -> Path:
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return Path(snapshot_download(source, revision=revision))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_release_files(source: Path, output: Path) -> None:
    excluded = {
        "model.safetensors",
        "cid_adapter.safetensors",
        "semantic-embedding.pt",
        "config.json",
        "cid_config.json",
        "modeling_lfm2_bidirectional.py",
        "configuration_cid.py",
        "modeling_cid.py",
    }
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.as_posix() in excluded:
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def convert(source: Path, output: Path) -> None:
    required = (
        source / "model.safetensors",
        source / "cid_adapter.safetensors",
        source / "semantic-embedding.pt",
        source / "config.json",
        source / "cid_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("legacy release is incomplete: " + ", ".join(missing))

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _copy_release_files(source, output)

    backbone_config: dict[str, Any] = json.loads(
        (source / "config.json").read_text(encoding="utf-8")
    )
    release_config: dict[str, Any] = json.loads(
        (source / "cid_config.json").read_text(encoding="utf-8")
    )
    semantic_state = torch.load(
        source / "semantic-embedding.pt",
        map_location="cpu",
        weights_only=False,
    )

    cid_config = build_unified_cid_config(
        backbone_config=backbone_config,
        adapter_config=release_config["adapter_config"],
        semantic_state=semantic_state,
        neural_contract_version=int(release_config["neural_contract_version"]),
        base_model=release_config.get("base_model"),
        release_name=release_config.get("release_name"),
    )
    cid_config.save_pretrained(output)

    backbone_state = load_file(source / "model.safetensors", device="cpu")
    cid_state = load_file(source / "cid_adapter.safetensors", device="cpu")
    unified_state = unified_state_from_legacy(backbone_state, cid_state, semantic_state)
    weight_path = output / "model.safetensors"
    save_file(unified_state, weight_path, metadata={"format": "pt"})

    (output / "configuration_cid.py").write_text(REMOTE_CONFIGURATION, encoding="utf-8")
    (output / "modeling_cid.py").write_text(REMOTE_MODELING, encoding="utf-8")

    semantic_metadata = dict(release_config.get("semantic_embedding_snapshot", {}))
    semantic_metadata.pop("sha256", None)
    semantic_metadata.update(
        {
            "file": "model.safetensors",
            "tensor_key": "semantic_embedding_weight",
        }
    )
    release_config["semantic_embedding_snapshot"] = semantic_metadata

    old_weights = release_config.get("weights", {})
    backbone_parameters = int(old_weights.get("backbone", {}).get("parameter_count", 0))
    cid_parameters = int(old_weights.get("cid_modules", {}).get("parameter_count", 0))
    total_parameters = int(
        old_weights.get("total_parameter_count", backbone_parameters + cid_parameters)
    )
    release_config["weights"] = {
        "layout": "unified-cid-model",
        "file": "model.safetensors",
        "sha256": _sha256(weight_path),
        "backbone_parameter_count": backbone_parameters,
        "cid_parameter_count": cid_parameters,
        "model_parameter_count": total_parameters,
        "semantic_snapshot_value_count": int(semantic_state["weight"].numel()),
        "tensor_count": len(unified_state),
    }
    release_config["format_version"] = max(int(release_config.get("format_version", 1)), 2)
    (output / "cid_config.json").write_text(
        json.dumps(release_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"converted {source} -> {output}")
    print(f"weights: {weight_path.stat().st_size} bytes")
    print(f"sha256: {release_config['weights']['sha256']}")
    print(f"tensors: {len(unified_state)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a legacy CID backbone+sidecar HF release to one CID weight file."
    )
    parser.add_argument("source", help="Legacy local directory or Hugging Face model repository")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--revision", default=None, help="Optional source Hugging Face revision")
    args = parser.parse_args()
    convert(_source_directory(args.source, args.revision), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
