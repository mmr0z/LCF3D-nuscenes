#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${LCF3D_CAMERA_CONFIG:-$REPO_ROOT/src/model_configs/nuscenes/faster_rcnn_lcf3d_nuscenes_camera.py}"
WORK_DIR="${LCF3D_CAMERA_WORK_DIR:-$REPO_ROOT/work_dirs/faster_rcnn_lcf3d_nuscenes_camera}"
NUSCENES_ROOT="${LCF3D_NUSCENES_ROOT:-$HOME/mmdetection3d/data/nuscenes}"
COCO_ROOT="${LCF3D_CAMERA_COCO_ROOT:-$REPO_ROOT/data/nuscenes_camera_coco}"
GPUS="${LCF3D_GPUS:-1}"
BATCH_SIZE="${LCF3D_CAMERA_BATCH_SIZE:-10}"
NUM_WORKERS="${LCF3D_CAMERA_NUM_WORKERS:-2}"

export LCF3D_NUSCENES_ROOT="$NUSCENES_ROOT"
export LCF3D_CAMERA_COCO_ROOT="$COCO_ROOT"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$REPO_ROOT/src/mmdetection3d:$REPO_ROOT/src:${PYTHONPATH:-}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in \
  "$COCO_ROOT/nuscenes_camera_train.json" \
  "$COCO_ROOT/nuscenes_camera_val.json" \
  "$NUSCENES_ROOT/samples"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required camera training path: $required" >&2
    echo "Run first: ./src/prepare_nuscenes_camera2d.sh" >&2
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
        'Activate a CUDA-capable env/node before camera training. '
        'Set LCF3D_ALLOW_CPU=1 only for config/data debugging.',
        file=sys.stderr)
    sys.exit(1)
PY
fi

train_args=()
if [[ "${LCF3D_AMP:-0}" == "1" ]]; then
  train_args+=("--amp")
fi
if [[ "${LCF3D_AUTO_SCALE_LR:-0}" == "1" ]]; then
  train_args+=("--auto-scale-lr")
fi
if [[ -n "${LCF3D_RESUME:-}" ]]; then
  train_args+=("--resume" "$LCF3D_RESUME")
fi

mim train mmdet "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --gpus "$GPUS" \
  --launcher none \
  "${train_args[@]}" \
  --cfg-options \
  "train_dataloader.batch_size=$BATCH_SIZE" \
  "train_dataloader.num_workers=$NUM_WORKERS" \
  "$@"
