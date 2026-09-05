#!/bin/bash
# RunPod training: full 7B multitask run on all 4 datasets (cholec80,
# cholect50, surg396k, heichole). The HeiChole LR sweep was removed.
#
#   bash runpod/train.sh                    # full 7B run (runpod_7b_config.yaml)
#   ... [--max_steps 500]                   # extra args passthrough
#
# Heavy state (checkpoints, HF cache, logs) lives on the shared Network Volume
# at $RUNPOD_VOLUME so pods can be terminated/restarted without losing work.

set -euo pipefail

REPO="${REPO:-/workspace}"
[ -d "$REPO" ] || { echo "ERROR: repo not found at $REPO (image bakes it at /workspace)" >&2; exit 1; }
cd "$REPO"

RUNPOD_VOLUME="${RUNPOD_VOLUME:-/runpod-volume}"
[ -d "$RUNPOD_VOLUME" ] || { echo "ERROR: Network Volume not mounted at $RUNPOD_VOLUME" >&2; exit 1; }

export HF_HOME="${HF_HOME:-$RUNPOD_VOLUME/hf-cache}"
export TOKENIZERS_PARALLELISM=false
# Fight CUDA fragmentation so the first optimizer.step() fits on 32GB VRAM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUNPOD_VOLUME/hf-cache" "$RUNPOD_VOLUME/logs" "$RUNPOD_VOLUME/checkpoints"

# Split files live on the network volume; the loader resolves data/splits
# against $REPO (/workspace). Symlink them in so dataset loading finds them.
DATA_SPLITS="$REPO/data/splits"
if [ ! -e "$DATA_SPLITS" ]; then
  mkdir -p "$(dirname "$DATA_SPLITS")"
  ln -s "$RUNPOD_VOLUME/data/splits" "$DATA_SPLITS"
  echo "linked $DATA_SPLITS -> $RUNPOD_VOLUME/data/splits"
fi

echo "=== job start $(date -u +%FT%TZ) host=$(hostname) repo=$REPO volume=$RUNPOD_VOLUME ==="

EXTRA=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) echo "ERROR: use the config baked into this script (configs/training/runpod_7b_config.yaml)" >&2; exit 1 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

python scripts/train_multitask.py \
  --config configs/training/runpod_7b_config.yaml \
  "${EXTRA[@]}"