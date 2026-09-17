from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import fsspec
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

from cid.model.ar import prepare_ar_tokenizer


SOURCE_SPECS = (
    {
        "name": "ultrax-web-en",
        "repo": "openbmb/UltraX-Preview",
        "pattern": "data/UltraX-Ultra-FineWeb/*.parquet",
        "column": "cleaned_content",
        "weight": 0.70,
    },
    {
        "name": "ultrafineweb-zh",
        "repo": "openbmb/Ultra-FineWeb",
        "pattern": "data/ultrafineweb_zh/*.parquet",
        "column": "content",
        "weight": 0.15,
    },
    {
        "name": "ultradata-code",
        "repo": "openbmb/UltraData-Code",
        "pattern": "data/UltraData-Code-L2/*/*.parquet",
        "column": "content",
        "weight": 0.10,
    },
    {
        "name": "ultradata-math",
        "repo": "openbmb/UltraData-Math",
        "pattern": "data/UltraData-Math-L2-preview/*.parquet",
        "column": "content",
        "weight": 0.05,
    },
)


def tokenize_source(
    *,
    spec: dict[str, object],
    tokenizer,
    output: Path,
    target_tokens: int,
    seed: int,
    batch_size: int,
) -> dict[str, object]:
    fs = HfFileSystem()
    repo = str(spec["repo"])
    pattern = str(spec["pattern"])
    paths = fs.glob(f"datasets/{repo}/{pattern}")
    random.Random(seed).shuffle(paths)
    if not paths:
        raise RuntimeError(f"no parquet files matched {repo}/{pattern}")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("tokenizer must define eos_token_id")

    written = 0
    documents = 0
    with output.open("wb") as handle:
        for hf_path in paths:
            remote = "hf://" + hf_path
            with fsspec.open(remote, "rb") as stream:
                parquet = pq.ParquetFile(stream)
                for record_batch in parquet.iter_batches(
                    batch_size=batch_size,
                    columns=[str(spec["column"])],
                ):
                    texts = [
                        value.as_py()
                        for value in record_batch.column(0)
                        if value.as_py() and value.as_py().strip()
                    ]
                    if not texts:
                        continue
                    encoded = tokenizer(
                        texts,
                        add_special_tokens=False,
                        padding=False,
                        truncation=False,
                    )["input_ids"]
                    for token_ids in encoded:
                        if not token_ids:
                            continue
                        values = np.asarray([*token_ids, int(eos)], dtype=np.uint32)
                        remaining = target_tokens - written
                        if remaining <= 0:
                            break
                        if len(values) > remaining:
                            values = values[:remaining]
                        values.tofile(handle)
                        written += len(values)
                        documents += 1
                    if written >= target_tokens:
                        break
            if written >= target_tokens:
                break
    return {
        "name": spec["name"],
        "repo": repo,
        "pattern": pattern,
        "column": spec["column"],
        "weight": spec["weight"],
        "file": output.name,
        "tokens": written,
        "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B-Base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.total_tokens <= 0 or args.sequence_length <= 0:
        raise ValueError("token budget and sequence length must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mask_token_id = prepare_ar_tokenizer(tokenizer)
    tokenizer.save_pretrained(output / "tokenizer")

    sources = []
    for index, spec in enumerate(SOURCE_SPECS):
        target = int(args.total_tokens * float(spec["weight"]))
        source_path = output / f"{spec['name']}.uint32.bin"
        source = tokenize_source(
            spec=spec,
            tokenizer=tokenizer,
            output=source_path,
            target_tokens=target,
            seed=args.seed + index * 997,
            batch_size=args.batch_size,
        )
        # Sampling is random-window based, so keeping exact target lengths is fine; no
        # per-source sequence-boundary truncation is required.
        sources.append(source)
        print(json.dumps(source, ensure_ascii=False), flush=True)

    manifest = {
        "format": "cid-diffusion-packed-v1",
        "base_model": args.model,
        "sequence_length": args.sequence_length,
        "mask_token_id": mask_token_id,
        "total_tokens": sum(int(source["tokens"]) for source in sources),
        "sources": sources,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
