from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cid.model.encoding import ILLaDATextEncoder
from cid.model.illada import ILLaDACIDConfig
from cid.model.lfm import LFMCIDAdapter
from cid.model.lfm_native import LFMNativeDisplayDecoder
from cid.model.tensors import CIDTensorBatch


PROMPTS = (
    ("greeting", "Hello!"),
    ("identity", "Who are you?"),
    ("definition", "What is a computer?"),
    ("code", "Write a Python function to reverse a string."),
    ("zh", "你好，请用中文介绍一下自己。"),
)


def make_batch(
    adapter: LFMCIDAdapter,
    prefix: torch.Tensor,
    display: torch.Tensor,
) -> CIDTensorBatch:
    device = prefix.device
    dtype = adapter.input_embeddings.weight.dtype
    slots = adapter.config.max_thought_slots
    width = adapter.d_model
    return CIDTensorBatch(
        thought_semantic=torch.zeros(1, slots, width, device=device, dtype=dtype),
        role_features=torch.zeros(1, slots, adapter.config.num_roles, device=device, dtype=dtype),
        uncertainty=torch.zeros(1, slots, 1, device=device, dtype=dtype),
        local_noise=torch.zeros(1, slots, 1, device=device, dtype=dtype),
        slot_occupancy=torch.zeros(1, slots, 1, device=device, dtype=dtype),
        prompt_ids=prefix,
        display_ids=display,
        display_noise=torch.ones(1, display.shape[1], 1, device=device, dtype=dtype),
        fact_memory=torch.empty(1, 0, width, device=device, dtype=dtype),
        percept_memory=torch.empty(1, 0, width, device=device, dtype=dtype),
        source_memory=torch.empty(1, 0, width, device=device, dtype=dtype),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--canvas", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = ILLaDACIDConfig(
        max_thought_slots=8,
        max_display_tokens=args.canvas + 1,
        display_canvas_tokens=args.canvas + 1,
        backbone_protocol="lfm-native",
        interaction_architecture="identity-preserving",
    )
    adapter = LFMCIDAdapter.from_pretrained(
        args.model, config=config, freeze_backbone=True, torch_dtype=torch.float32
    ).to(device).eval()
    text_encoder = ILLaDATextEncoder(adapter, tokenizer)
    decoder = LFMNativeDisplayDecoder(
        tokenizer, mask_id=adapter.mask_token_id, eos_id=adapter.eos_token_id
    )
    rows = []
    with torch.inference_mode():
        for name, question in PROMPTS:
            prefix = text_encoder.tokenize_prompt(question, protocol="lfm-native")
            native_prefix = tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                ), return_tensors="pt"
            ).input_ids.to(device)
            assert torch.equal(prefix, native_prefix)
            native = torch.full(
                (1, args.canvas), int(tokenizer.mask_token_id),
                dtype=torch.long, device=device
            )
            cid = torch.cat((
                native.clone(),
                torch.full((1, 1), int(tokenizer.eos_token_id),
                           dtype=torch.long, device=device)
            ), dim=1)
            peak_diff = 0.0
            divergence_step = None
            for step in range(args.canvas):
                direct = adapter.backbone(
                    input_ids=torch.cat((prefix, native), dim=1),
                    attention_mask=torch.ones(
                        (1, prefix.shape[1] + args.canvas),
                        dtype=torch.long, device=device
                    ),
                ).logits[:, prefix.shape[1]:, :]
                bridged = adapter(make_batch(adapter, prefix, cid)).display_logits
                error = float((bridged[:, :-1, :direct.shape[-1]] - direct).abs().max())
                peak_diff = max(peak_diff, error)
                if error > 2e-4 and divergence_step is None:
                    divergence_step = step
                if bool(native.eq(tokenizer.mask_token_id).any()):
                    mask_positions = native[0].eq(tokenizer.mask_token_id).nonzero().flatten()
                    c = direct[0, mask_positions, :len(tokenizer)].float().clone()
                    for special in tokenizer.all_special_ids:
                        if special != tokenizer.eos_token_id and special < c.shape[-1]:
                            c[:, int(special)] = -torch.inf
                    confidence, guesses = c.softmax(-1).max(-1)
                    best = int(confidence.argmax())
                    native[0, mask_positions[best]] = guesses[best]
                cid = decoder.refine(cid, bridged)
                if not cid[0, :-1].eq(tokenizer.mask_token_id).any():
                    break
                if cid[0, :-1].eq(tokenizer.eos_token_id).any():
                    break
            native_text = tokenizer.decode(native[0], skip_special_tokens=True)
            native_text = native_text.split("[/Answer]")[0].strip()
            end = cid[0].eq(tokenizer.eos_token_id).nonzero().flatten()[0].item()
            cid_text = tokenizer.decode(cid[0, :end], skip_special_tokens=True).strip()
            row = {
                "task": name,
                "question": question,
                "native": native_text,
                "cid": cid_text,
                "same_text": native_text == cid_text,
                "native_markup_leak": "[/Answer]" in cid_text,
                "cid_unresolved_before_eos": int(
                    cid[0, :end].eq(tokenizer.mask_token_id).sum()
                ),
                "iterations": step + 1,
                "max_native_logit_error": peak_diff,
                "first_logit_divergence_step": divergence_step,
            }
            rows.append(row)
            print("CHECK", json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "model": args.model,
        "n_tests": len(rows),
        "matches": sum(row["same_text"] for row in rows),
        "maximum_logit_error": max(row["max_native_logit_error"] for row in rows),
        "native_markup_leaks": sum(row["native_markup_leak"] for row in rows),
        "rows": rows,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({
        key: value for key, value in report.items() if key != "rows"
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
