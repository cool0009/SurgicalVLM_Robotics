#!/bin/bash
# One-time RunPod volume bootstrap. Replaces the old setup_volume.sh +
# extract_frames.sh pair: extracts the data bundle if not present, verifies
# every image the JSONL references, and auto-runs frame extraction when images
# are missing (raw-video datasets) so the volume becomes training-ready.
#
#   bash runpod/setup.sh
#
# Prereq: data.tar uploaded to the volume (RunPod dashboard > Network Volumes),
# or a data/ tree already present on the volume.

set -euo pipefail

REPO="${REPO:-/workspace}"
[ -d "$REPO" ] || { echo "ERROR: repo not found at $REPO (image bakes it at /workspace)" >&2; exit 1; }
cd "$REPO"

RUNPOD_VOLUME="${RUNPOD_VOLUME:-/runpod-volume}"
[ -d "$RUNPOD_VOLUME" ] || { echo "ERROR: Network Volume not mounted at $RUNPOD_VOLUME" >&2; exit 1; }

DATA="$RUNPOD_VOLUME/data"
SPLITS_DIR="$DATA/splits"
OUTPUT_ROOT="$DATA/frames"

echo "=== volume setup start $(date -u +%FT%TZ) host=$(hostname) volume=$RUNPOD_VOLUME ==="

# -- 1) Extract the data bundle (skip if already present) ----------------------
if [ -d "$DATA/processed/vlm_jsonl" ]; then
  echo "data bundle already present, reusing it."
else
  TARBALL="${DATA_TARBALL:-}"
  if [ -z "$TARBALL" ]; then
    for cand in "$RUNPOD_VOLUME/data.tar" "$RUNPOD_VOLUME"/*.tar "$REPO/data.tar"; do
      [ -f "$cand" ] && TARBALL="$cand" && break
    done
  fi
  if [ -z "$TARBALL" ]; then
    echo "ERROR: no data.tar on the volume and no data/processed/vlm_jsonl." >&2
    echo "Upload data.tar via the RunPod dashboard, then re-run." >&2
    exit 1
  fi
  echo "extracting $TARBALL -> $RUNPOD_VOLUME"
  tar -xf "$TARBALL" -C "$RUNPOD_VOLUME"
fi

# -- 2) Verify every jsonl-referenced image exists on the volume ---------------
echo "=== check_data_bundle.py ==="
set +e
python scripts/check_data_bundle.py \
  --root "$DATA" \
  --jsonl-dir "$DATA/processed/vlm_jsonl" \
  --workers 8
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  echo "PASS: all vlm_jsonl images present on the volume."
elif [ "$RC" -eq 1 ]; then
  echo "NOTE: some images missing - extracting frames from raw data ..."
  # HeiChole: raw videos -> uniform 1 fps frames, using the train/val/test splits.
  mkdir -p "$OUTPUT_ROOT/heichole"
  for SPLIT in train val test; do
    python scripts/extract_frames.py \
      --data-root "$DATA/raw" \
      --output-root "$OUTPUT_ROOT" \
      --dataset heichole \
      --splits-dir "$SPLITS_DIR" \
      --split "$SPLIT" \
      --strategy uniform \
      --fps 1 \
      --workers 8
  done
  # CholecT50: pre-extracted PNGs -> jpg (jsonl references .jpg).
  mkdir -p "$OUTPUT_ROOT/cholect50"
  python - "$DATA/raw" "$OUTPUT_ROOT" <<'PY'
import sys
from pathlib import Path
import cv2

raw = Path(sys.argv[1]) / "CholecT50" / "videos"
out = Path(sys.argv[2]) / "cholect50"
converted = failed = 0
for src_dir in sorted(raw.iterdir()):
    if not src_dir.is_dir():
        continue
    dst_dir = out / src_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    for png in sorted(src_dir.glob("*.png")):
        dst = dst_dir / f"frame_{png.stem}.jpg"
        if dst.exists():
            continue
        img = cv2.imread(str(png))
        if img is None:
            failed += 1
            continue
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        converted += 1
print(f"CholecT50: converted={converted} failed={failed}")
PY
  echo "frames written under $OUTPUT_ROOT; re-running verify ..."
  python scripts/check_data_bundle.py \
    --root "$DATA" \
    --jsonl-dir "$DATA/processed/vlm_jsonl" \
    --workers 8
else
  echo "FAIL: check_data_bundle hit hard errors." >&2
  exit 1
fi

echo "=== Volume ready. Next: bash runpod/train.sh (or train.sh --lr <lr>) ==="