from __future__ import annotations

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cid.model.modeling_cid_diffusion import CIDDiffusionForMaskedLM


def tiny_config():
    config = transformers.LlamaConfig(
        vocab_size=33,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        tie_word_embeddings=False,
    )
    config.mask_token_id = 32
    config.use_cache = False
    config.cid_diffusion_base = True
    config.diffusion_objective = "masked-diffusion"
    config.diffusion_attention = "bidirectional"
    return config


def test_diffusion_hf_forward_is_bidirectional() -> None:
    torch.manual_seed(17)
    model = CIDDiffusionForMaskedLM(tiny_config()).eval()
    first = torch.tensor([[2, 3, 4]])
    second = torch.tensor([[2, 5, 4]])

    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits

    assert not torch.allclose(first_logits[:, 0], second_logits[:, 0])


def test_diffusion_hf_denoise_resolves_only_masks() -> None:
    torch.manual_seed(19)
    model = CIDDiffusionForMaskedLM(tiny_config()).eval()
    inputs = torch.tensor([[7, 32, 8, 32]])
    output = model.denoise(inputs, steps=2)

    assert output.shape == inputs.shape
    assert output[0, 0] == 7
    assert output[0, 2] == 8
    assert not output.eq(32).any()


def test_diffusion_hf_generate_appends_resolved_canvas() -> None:
    torch.manual_seed(23)
    model = CIDDiffusionForMaskedLM(tiny_config()).eval()
    prompt = torch.tensor([[7, 8, 9]])

    output = model.generate(
        prompt,
        max_new_tokens=4,
        diffusion_steps=4,
        block_length=2,
    )

    assert output.shape == (1, 7)
    assert torch.equal(output[:, :3], prompt)
    assert not output.eq(32).any()


def test_remote_code_round_trip(tmp_path: Path) -> None:
    model = CIDDiffusionForMaskedLM(tiny_config()).eval()
    model.config.architectures = ["CIDDiffusionForMaskedLM"]
    model.config.auto_map = {
        "AutoModelForCausalLM": "modeling_cid_diffusion.CIDDiffusionForMaskedLM"
    }
    model.save_pretrained(tmp_path, safe_serialization=True)

    source = Path(__file__).resolve().parents[1] / "src/cid/model/modeling_cid_diffusion.py"
    shutil.copy2(source, tmp_path / "modeling_cid_diffusion.py")

    restored = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    assert restored.__class__.__name__ == "CIDDiffusionForMaskedLM"
    assert hasattr(restored, "diffusion_generate")

    ids = torch.tensor([[7, 32, 8]])
    with torch.no_grad():
        logits = restored(ids).logits
    assert logits.shape == (1, 3, 33)
    assert torch.isfinite(logits).all()
