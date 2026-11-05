from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_native_reply_stop_tokenization_matches_runtime() -> None:
    model_path = os.environ.get("CID_LFM_NATIVE_MODEL_DIR")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("requires the locally cached LFM350 tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    stop = tokenizer.encode("[/Answer]", add_special_tokens=False)
    assert stop
    for reply in ("37.", "Yes.", "The answer is 131 ms.", "r55", "The result is false."):
        ids = tokenizer.encode(reply + "\n[/Answer]", add_special_tokens=False)
        assert ids[-len(stop):] == stop
