#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONFIG="${LCF3D_CAMERA_CONFIG:-$REPO_ROOT/src/model_configs/nuscenes/faster_rcnn_lcf3d_nuscenes_camera.py}"
CHECKPOINT="${LCF3D_CAMERA_CHECKPOINT:-$REPO_ROOT/work_dirs/faster_rcnn_lcf3d_nuscenes_camera_nuimages_ft6_bs10/best_coco_bbox_mAP_iter_85134.pth}"
NUSCENES_ROOT="${LCF3D_NUSCENES_ROOT:-$HOME/mmdetection3d/data/nuscenes}"
COCO_ROOT="${LCF3D_CAMERA_COCO_ROOT:-$REPO_ROOT/data/nuscenes_camera_coco}"
OUTPUT_ROOT="${LCF3D_CAMERA_TEST_ROOT:-$REPO_ROOT/work_dirs/faster_rcnn_lcf3d_nuscenes_camera_test}"
DEVICE="${LCF3D_DEVICE:-cuda:0}"
PYTHON_BIN="${LCF3D_PYTHON:-python}"
BATCH_SIZE="${LCF3D_CAMERA_TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${LCF3D_CAMERA_TEST_NUM_WORKERS:-2}"
PERSISTENT_WORKERS=true
if [[ "$NUM_WORKERS" == "0" ]]; then
  PERSISTENT_WORKERS=false
fi
RUN_NAME="${LCF3D_RUN_NAME:-faster_rcnn_camera_best}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Activate the LCF3D environment or set LCF3D_PYTHON=/path/to/python." >&2
  exit 1
fi

for required in \
  "$CONFIG" \
  "$CHECKPOINT" \
  "$COCO_ROOT/nuscenes_camera_val.json" \
  "$NUSCENES_ROOT/samples"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required camera test path: $required" >&2
    exit 1
  fi
done

if [[ "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
  export CUDA_VISIBLE_DEVICES="${LCF3D_CUDA_VISIBLE_DEVICES:-${BASH_REMATCH[1]}}"
elif [[ "$DEVICE" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

if [[ "$DEVICE" == cuda:* && "${LCF3D_ALLOW_CPU:-0}" != "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print('CUDA is unavailable. Run on a GPU node or set LCF3D_DEVICE=cpu and LCF3D_ALLOW_CPU=1.', file=sys.stderr)
    sys.exit(1)
PY
fi

mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/model_config.py"

export LCF3D_NUSCENES_ROOT="$NUSCENES_ROOT"
export LCF3D_CAMERA_COCO_ROOT="$COCO_ROOT"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$REPO_ROOT/src/mmdetection3d:$REPO_ROOT/src:${PYTHONPATH:-}"

"$PYTHON_BIN" "$REPO_ROOT/src/mmdetection3d/tools/test.py" \
  "$CONFIG" \
  "$CHECKPOINT" \
  --work-dir "$RUN_DIR" \
  --cfg-options \
  "test_dataloader.batch_size=$BATCH_SIZE" \
  "test_dataloader.num_workers=$NUM_WORKERS" \
  "test_dataloader.persistent_workers=$PERSISTENT_WORKERS" \
  "test_evaluator.outfile_prefix=$RUN_DIR/predictions"

"$PYTHON_BIN" "$REPO_ROOT/src/analyze_thesis_metrics.py" \
  "$RUN_DIR" \
  --output-dir "$RUN_DIR/thesis_metrics" \
  --labels "$RUN_NAME"

echo "Camera test and thesis metrics: $RUN_DIR"
