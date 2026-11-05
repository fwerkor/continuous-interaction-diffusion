from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from validate_native_display_data import validate_row


def assemble(trajectories: Path, reviews: Path, output: Path) -> dict[str, object]:
    if trajectories.resolve() == output.resolve():
        raise ValueError("output must not overwrite the original CID dataset")
    annotations: dict[str, dict] = {}
    with reviews.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            identifier = item["example_id"]
            if identifier in annotations:
                raise ValueError(f"duplicate review for {identifier}")
            if (
                item.get("verified") is not True
                or not isinstance(item.get("reviewer"), str)
                or not item["reviewer"].strip()
            ):
                raise ValueError(f"{identifier}: verified review with provenance is required")
            annotations[identifier] = item

    output.parent.mkdir(parents=True, exist_ok=True)
    path = None
    written = 0
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=".native-full-", delete=False
        ) as target:
            path = Path(target.name)
            with trajectories.open(encoding="utf-8") as source:
                for number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    identifier = item["example_id"]
                    review = annotations.pop(identifier, None)
                    if review is None:
                        raise ValueError(f"{identifier}: no verified full reply")
                    metadata = dict(item.get("metadata", {}))
                    if "native_full_display" in metadata:
                        raise ValueError(f"{identifier}: output metadata already annotated")
                    metadata["native_full_display"] = review["full_reply"]
                    metadata["native_full_display_verified"] = True
                    metadata["native_full_display_reviewer"] = review["reviewer"]
                    metadata["native_full_display_semantic_paraphrase"] = (
                        review.get("semantic_paraphrase", False) is True
                    )
                    metadata["native_full_display_by_step"] = review.get("by_step", {})
                    item["metadata"] = metadata
                    errors = validate_row(item)
                    if errors:
                        raise ValueError(f"{identifier}: {'; '.join(errors)}")
                    data = (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    target.write(data)
                    digest.update(data)
                    written += 1
        if annotations:
            raise ValueError(f"{len(annotations)} review records had no matching CID example")
        if not written:
            raise ValueError("empty CID dataset")
        os.replace(path, output)
        path = None
        return {"rows": written, "sha256": digest.hexdigest(), "output": str(output)}
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge independently verified, grounded full replies into CID trajectories"
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--verified-replies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        assemble(args.trajectories, args.verified_replies, args.output),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
