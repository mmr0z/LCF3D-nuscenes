#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUSCENES_ROOT="${LCF3D_NUSCENES_ROOT:-$HOME/mmdetection3d/data/nuscenes}"
OUT_DIR="${LCF3D_CAMERA_COCO_ROOT:-$REPO_ROOT/data/nuscenes_camera_coco}"

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$REPO_ROOT/src/mmdetection3d:$REPO_ROOT/src:${PYTHONPATH:-}"

for required in \
  "$NUSCENES_ROOT/nuscenes_infos_train.pkl" \
  "$NUSCENES_ROOT/nuscenes_infos_val.pkl" \
  "$NUSCENES_ROOT/samples/CAM_FRONT" \
  "$NUSCENES_ROOT/samples/CAM_FRONT_LEFT" \
  "$NUSCENES_ROOT/samples/CAM_FRONT_RIGHT" \
  "$NUSCENES_ROOT/samples/CAM_BACK" \
  "$NUSCENES_ROOT/samples/CAM_BACK_LEFT" \
  "$NUSCENES_ROOT/samples/CAM_BACK_RIGHT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required nuScenes camera path: $required" >&2
    exit 1
  fi
done

extra_args=()
if [[ -n "${LCF3D_MAX_SAMPLES:-}" ]]; then
  extra_args+=("--max-samples" "$LCF3D_MAX_SAMPLES")
fi

python "$REPO_ROOT/src/dataset_utils/nuscenes_infos_to_coco_2d.py" \
  --nuscenes-root "$NUSCENES_ROOT" \
  --out-dir "$OUT_DIR" \
  "${extra_args[@]}" \
  "$@"
