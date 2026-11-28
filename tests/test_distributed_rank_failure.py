from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cid.cli import _fail_distributed_rank


def test_single_rank_failure_does_not_force_exit():
    class Dist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 1

    _fail_distributed_rank(Dist(), RuntimeError("ordinary exception"))


def test_multi_rank_failure_reports_original_error_and_exits(tmp_path):
    error_file = tmp_path / "elastic-error.json"
    code = """
import os
from cid.cli import _fail_distributed_rank
class Dist:
    def is_initialized(self): return True
    def get_world_size(self): return 4
    def get_rank(self): return 2
try:
    raise RuntimeError("injected rank-local rollout failure")
except BaseException as error:
    _fail_distributed_rank(Dist(), error)
raise AssertionError("failed rank must not enter collective cleanup")
"""
    import os
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TORCHELASTIC_ERROR_FILE": str(error_file)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "CID_DISTRIBUTED_RANK_FAILURE rank=2" in result.stderr
    assert "RuntimeError: injected rank-local rollout failure" in result.stderr
    assert "injected rank-local rollout failure" in error_file.read_text()
