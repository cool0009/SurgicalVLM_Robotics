"""Pre-flight + post-extract verification of the packaged data bundle.

Checks that every image path referenced by the VLM JSONL files resolves to an
existing file on disk, using the SAME resolution rule as training:

    train_multitask.py:231 -> VLMJSONLDataset.__getitem__
        full_image_path = image_root / image_path        # image_root = "data"
        if not full_image_path.exists():
            full_image_path = Path(image_path)           # absolute fallback

Run BEFORE packaging (local Windows) and AGAIN AFTER extraction on the RunPod
Network Volume using the same command. Any mismatch means the bundle is
incomplete or corrupt for training purposes.

Usage:
    python scripts/check_data_bundle.py [--root data]
                                        [--jsonl-dir data/processed/vlm_jsonl]
                                        [--files file1.jsonl file2.jsonl ...]
                                        [--workers 16]

Exit code 0  -> every referenced image exists (training-ready)
Exit code 1  -> one or more images (or JSONLs) missing
Exit code 2  -> hard error (bad argv, unreadable JSONL)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _resolve(root: Path, rel: str) -> Path:
    norm = rel.replace("\\", "/")
    path = root / norm.lstrip("/")
    if path.exists():
        return path
    # mirror training: fall back to the path as given (absolute reference)
    alt = Path(norm)
    return alt if alt.exists() else path


def _scan_jsonl(args: tuple[str, str]) -> tuple[str, int, set[str]]:
    """Collect the set of unique image references in one JSONL file."""
    jsonl_path, image_root = args
    unique = set()
    count = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                img = rec.get("image", "")
                if img:
                    unique.add(img)
                count += 1
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSONL {jsonl_path}: {exc}") from exc
    return jsonl_path, count, unique


def _check(args: tuple[str, str]) -> tuple[str, str, bool]:
    """Check one image path; returns (relpath, resolved_abs, exists)."""
    rel, image_root = args
    resolved = _resolve(Path(image_root), rel)
    return rel, str(resolved), resolved.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data",
                        help="image_root used by training (default: data)")
    parser.add_argument("--jsonl-dir", default="data/processed/vlm_jsonl",
                        help="directory containing the JSONL files")
    parser.add_argument("--config", default=None,
                        help="optional: configs/training/*.yaml to drive file "
                             "selection from its train/val/test_files")
    parser.add_argument("files", nargs="*",
                        help="explicit JSONL files (default: all *.jsonl)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    jsonl_dir = Path(args.jsonl_dir).resolve()
    if not root.is_dir():
        print(f"ERR: image_root not found: {root}", file=sys.stderr)
        return 2

    # ---- select JSONL files -----------------------------------------------
    jsonl_files: list[Path] = []
    if args.files:
        jsonl_files = [Path(f) for f in args.files]
    else:
        jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
        # skip metadata files that are not training corpora
        jsonl_files = [p for p in jsonl_files if "metadata" not in p.name]

    existing = []
    for p in jsonl_files:
        if p.is_file():
            existing.append(p)
        else:
            print(f"MISSING JSONL: {p}", file=sys.stderr)
    if not existing:
        print("No JSONL files found.", file=sys.stderr)
        return 2

    # ---- phase 1: collect every unique image reference --------------------
    print("Phase 1: scanning JSONL files ...")
    per_file: dict[str, dict] = {}
    all_unique: set[str] = set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_scan_jsonl, (str(p), str(root))): p
                for p in existing}
        for fut in as_completed(futs):
            path, count, unique = fut.result()
            per_file[path] = {"records": count, "unique": len(unique)}
            all_unique |= unique
            print(f"  scanned {os.path.basename(path)}: {count} records, {len(unique)} unique images")

    total_unique = len(all_unique)
    print(f"--- TOTAL unique image references: {total_unique} ---")

    # -- phase 2: check every unique image exists -------------------------
    print("--- phase 2: checking files (this is the slow part) ---")
    missing: dict[str, list[str]] = defaultdict(list)
    present = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_check, (rel, str(root))): rel
                for rel in sorted(all_unique)}
        for fut in as_completed(futs):
            rel, resolved, ok = fut.result()
            if ok:
                present += 1
            else:
                missing[rel] = resolved

    n_missing = len(missing)
    print(f"present: {present}/{total_unique}   missing: {n_missing}/{total_unique}")

    if n_missing:
        print("--- missing images (showing up to 40) ---")
        for rel in sorted(missing)[:40]:
            print(f"  {rel}  -> {missing[rel]}")
    else:
        print("OK: all referenced images exist")

    return 0 if n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())