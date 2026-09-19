from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open


MODEL_FILE = Path(__file__).resolve().parents[1] / "src/cid/model/modeling_cid_diffusion.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    return parser.parse_args()


def update_json(path: Path, update) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    update(data)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def repair_safetensors_parameter_count(release_dir: Path) -> int | None:
    index_path = release_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    total_parameters = 0
    for shard_path in sorted(release_dir.glob("model-*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                parameter_count = 1
                for dimension in handle.get_slice(tensor_name).get_shape():
                    parameter_count *= dimension
                total_parameters += parameter_count
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.setdefault("metadata", {})["total_parameters"] = total_parameters
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return total_parameters


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir).resolve()
    config_path = release_dir / "config.json"
    tokenizer_config_path = release_dir / "tokenizer_config.json"
    diffusion_config_path = release_dir / "diffusion_config.json"
    if not config_path.is_file() or not diffusion_config_path.is_file():
        raise FileNotFoundError(
            "release directory must already contain config.json and diffusion_config.json"
        )

    repair_safetensors_parameter_count(release_dir)
    shutil.copy2(MODEL_FILE, release_dir / "modeling_cid_diffusion.py")

    def patch_config(config: dict[str, object]) -> None:
        config["architectures"] = ["CIDDiffusionForMaskedLM"]
        auto_map = dict(config.get("auto_map") or {})
        auto_map["AutoModelForCausalLM"] = (
            "modeling_cid_diffusion.CIDDiffusionForMaskedLM"
        )
        config["auto_map"] = auto_map
        config["use_cache"] = False
        config["cid_diffusion_base"] = True
        config["diffusion_objective"] = "masked-diffusion"
        config["diffusion_attention"] = "bidirectional"

    update_json(config_path, patch_config)

    if tokenizer_config_path.is_file():
        update_json(
            tokenizer_config_path,
            lambda config: config.__setitem__("fix_mistral_regex", True),
        )

    generation_config = {
        "_from_model_config": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "do_sample": False,
    }
    (release_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2) + "\n",
        encoding="utf-8",
    )

    diffusion = json.loads(diffusion_config_path.read_text(encoding="utf-8"))
    diffusion.setdefault("mask_ratio_range", [0.001, 1.0])
    diffusion["hf_loader"] = "CIDDiffusionForMaskedLM"
    diffusion["hf_auto_model"] = "AutoModelForCausalLM"
    diffusion["requires_trust_remote_code"] = True
    diffusion_config_path.write_text(
        json.dumps(diffusion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
