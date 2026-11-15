from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch import nn

from cid.model.illada import _chunked_lfm2_mlp_forward


class SmallLFM2MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(8, 24, bias=False, dtype=torch.float64)
        self.w3 = nn.Linear(8, 24, bias=False, dtype=torch.float64)
        self.w2 = nn.Linear(24, 8, bias=False, dtype=torch.float64)
        self._cid_mlp_chunk_size = 32
        self._cid_mlp_chunk_min_tokens = 256
        self.forward = MethodType(_chunked_lfm2_mlp_forward, self)


@pytest.mark.parametrize("seq_length,expected_calls", [(64, 1), (256, 1), (320, 10)])
def test_lfm2_adaptive_chunking_matches_unchunked_autograd(
    seq_length: int, expected_calls: int
) -> None:
    torch.manual_seed(57)
    module = SmallLFM2MLP()
    x = torch.randn(1, seq_length, 8, dtype=torch.float64, requires_grad=True)
    full_x = x.detach().clone().requires_grad_(True)
    calls = []
    handle = module.w1.register_forward_hook(lambda _, args, __: calls.append(args[0].shape[1]))
    actual = module(x)
    handle.remove()
    baseline = module.w2(torch.nn.functional.silu(module.w1(full_x)) * module.w3(full_x))
    torch.testing.assert_close(actual, baseline, rtol=1e-12, atol=1e-12)
    actual.square().sum().backward()
    baseline.square().sum().backward()
    torch.testing.assert_close(x.grad, full_x.grad, rtol=1e-12, atol=1e-12)
    assert len(calls) == expected_calls


def test_cuda_cache_reclamation_only_when_low_on_free_memory(monkeypatch) -> None:
    from types import SimpleNamespace

    from cid.model.training import CIDTrainer

    trainer = object.__new__(CIDTrainer)
    trainer._cuda_cache_release_min_free_bytes = 1024**3
    trainer.tensorizer = SimpleNamespace(
        text_encoder=SimpleNamespace(device=torch.device("cuda", 0))
    )
    reclaimed = []
    free = [512 * 1024**2]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (free[0], 24 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 8 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 6 * 1024**3)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: reclaimed.append(True))

    trainer._maybe_release_cuda_cache()
    assert len(reclaimed) == 1
    free[0] = 2 * 1024**3
    trainer._maybe_release_cuda_cache()
    assert len(reclaimed) == 1
    trainer._cuda_cache_release_min_free_bytes = 0
    free[0] = 0
    trainer._maybe_release_cuda_cache()
    assert len(reclaimed) == 1
