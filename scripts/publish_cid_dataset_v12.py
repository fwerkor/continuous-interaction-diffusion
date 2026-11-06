"""Publish only the CID-Dataset v1.2 language extension, preserving v1.1 blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="fwerkor/CID-Dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    root = args.dataset_dir.resolve()
    release_path = root / "RELEASE-MANIFEST-v1.2.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    additions = release["additions"]["language_sft"]
    expected = {
        additions["train"]: additions["train_sha256"],
        additions["validation"]: additions["validation_sha256"],
    }
    for relative, expected_sha in expected.items():
        actual = sha256(root / relative)
        if actual != expected_sha:
            raise ValueError(f"{relative}: SHA256 mismatch ({actual})")
    manifest_path = root / additions["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    anonymous = HfApi(token=False)
    repo = anonymous.repo_info(args.repo_id, repo_type="dataset")
    tree = {
        item.path: item
        for item in anonymous.list_repo_tree(
            args.repo_id,
            repo_type="dataset",
            recursive=True,
        )
    }
    base_path = release["base"]["path"]
    base_entry = tree.get(base_path)
    base_lfs = getattr(base_entry, "lfs", None)
    if base_lfs is None or base_lfs.sha256 != release["base"]["sha256"]:
        raise ValueError("HF v1.1 base is missing or its LFS SHA256 has changed")

    release_notes = (root / "README.md").read_text(encoding="utf-8")
    if not release_notes.startswith("# CID-Dataset v1.2"):
        raise ValueError("expanded release README does not describe v1.2")
    announcement = "\n\n" + release_notes.replace("# CID-Dataset v1.2", "## CID-Dataset v1.2", 1)
    remote_readme = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename="README.md",
            repo_type="dataset",
            token=False,
        )
    ).read_text(encoding="utf-8")
    if "## CID-Dataset v1.2" in remote_readme:
        raise ValueError("v1.2 announcement already present; verify remote release before retrying")
    expected_uploads = [*expected, additions["manifest"], "RELEASE-MANIFEST-v1.2.json"]
    collisions = [item for item in expected_uploads if item in tree]
    if collisions:
        raise ValueError(f"v1.2 remote paths already exist: {collisions}")

    print(f"repo={repo.id} base_sha256={base_lfs.sha256} unchanged")
    print(f"new_paths={expected_uploads} expected_language_examples={additions['examples']}")
    if args.dry_run:
        print("DRY RUN: no remote changes")
        return

    api = HfApi()
    print(f"publishing_as={api.whoami()['name']}")
    operations = [
        CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(root / relative))
        for relative in expected_uploads
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=(remote_readme + announcement).encode("utf-8"),
        )
    )
    commit = api.create_commit(
        args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="dataset: publish CID-Dataset v1.2 language_sft",
        parent_commit=repo.sha,
    )
    print(f"commit={commit.oid} url={commit.commit_url}")
    try:
        api.create_tag(
            args.repo_id,
            tag="v1.2",
            revision=commit.oid,
            repo_type="dataset",
        )
        print("tag=v1.2")
    except Exception as exc:
        print(f"tag_not_created={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
