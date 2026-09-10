#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUSCENES_ROOT="${LCF3D_NUSCENES_ROOT:-$HOME/mmdetection3d/data/nuscenes}"
CFG="${LCF3D_FUSION_CFG:-$REPO_ROOT/src/configs/lcf3d_nuscenes_authors_local.json}"
OUT_DIR="${LCF3D_EVAL_OUT:-$REPO_ROOT/work_dirs/lcf3d_nuscenes_authors_eval}"
DEVICE="${LCF3D_DEVICE:-cuda:0}"

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$REPO_ROOT/src/mmdetection3d:$REPO_ROOT/src:${PYTHONPATH:-}"

python "$REPO_ROOT/src/eval_nuscenes.py" \
  -output_dir "$OUT_DIR" \
  -nuscenes_root_path "$NUSCENES_ROOT" \
  -late_fusion_config "$CFG" \
  -device "$DEVICE" \
  --version v1.0-trainval \
  --eval_set val \
  --eval_version detection_cvpr_2019 \
  --sweep_num 10 \
  --use_intensity \
  --cameras CAM_FRONT_LEFT,CAM_FRONT,CAM_FRONT_RIGHT,CAM_BACK_LEFT,CAM_BACK,CAM_BACK_RIGHT \
  "$@"
