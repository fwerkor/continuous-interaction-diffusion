from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assemble_la_teacher_drafts.py"


def _run_assembler(tmp_path: Path, lengths: list[int]) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "teacher.jsonl"
    source.write_text(
        "".join(
            json.dumps({
                "prompt": f"question-{i}",
                "answer": f"teacher response {i}",
                "answer_tokens": length,
                "language": "en",
            }) + "\n"
            for i, length in enumerate(lengths)
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable, str(SCRIPT), "--sources", str(source),
            "--output", str(tmp_path / "merged.jsonl"),
            "--min-examples", str(len(lengths)),
        ],
        check=False, capture_output=True, text=True,
    )


def test_teacher_assembler_rejects_canvas_length_collapse(tmp_path: Path) -> None:
    result = _run_assembler(tmp_path, [59] * 20)
    assert result.returncode != 0
    assert "collapsed" in result.stderr or "concentrated" in result.stderr


def test_teacher_assembler_accepts_diverse_stopping_positions(tmp_path: Path) -> None:
    result = _run_assembler(
        tmp_path, [12, 16, 20, 24, 28, 32, 40, 48, 64, 72, 96, 128]
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "merged.manifest.json").read_text())
    assert report["count"] == 12
    assert report["length_short"] == 3
    assert report["length_long"] == 4
