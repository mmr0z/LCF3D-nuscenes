#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/src/model_configs/nuscenes/frustum_pointnet_lcf3d_nuscenes.py"
WORK_DIR="$REPO_ROOT/work_dirs/frustum_pointnet_lcf3d_nuscenes"
FRUSTUM_ROOT="${LCF3D_FRUSTUM_ROOT:-$REPO_ROOT/data/frustum_nuscenes_compact}"
BATCH_SIZE="${LCF3D_FRUSTUM_BATCH_SIZE:-64}"

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$REPO_ROOT/src/mmdetection3d:$REPO_ROOT/src:${PYTHONPATH:-}"

for required in \
  "$FRUSTUM_ROOT/nusc_frustum_info_train.pkl" \
  "$FRUSTUM_ROOT/nusc_frustum_info_val.pkl"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing frustum dataset file: $required" >&2
    echo "Run first: ./src/prepare_nuscenes_frustum.sh" >&2
    exit 1
  fi
done

if [[ "${LCF3D_ALLOW_CPU:-0}" != "1" ]]; then
  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print(
        'CUDA is not available to PyTorch in this shell. '
        'Use `conda activate lcf3d-cu128`, then rerun this script. '
        'Set LCF3D_ALLOW_CPU=1 only for config/data debugging.',
        file=sys.stderr)
    sys.exit(1)
PY
fi

python "$REPO_ROOT/src/mmdetection3d/tools/train.py" \
  "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --cfg-options \
  "data_root=$FRUSTUM_ROOT/" \
  "train_dataloader.batch_size=$BATCH_SIZE" \
  "train_dataloader.dataset.dataset.ann_file=$FRUSTUM_ROOT/nusc_frustum_info_train.pkl" \
  "val_dataloader.dataset.ann_file=$FRUSTUM_ROOT/nusc_frustum_info_val.pkl" \
  "test_dataloader.dataset.ann_file=$FRUSTUM_ROOT/nusc_frustum_info_val.pkl" \
  "$@"
