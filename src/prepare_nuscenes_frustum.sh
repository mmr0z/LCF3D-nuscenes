#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUSCENES_ROOT="${LCF3D_NUSCENES_ROOT:-$HOME/mmdetection3d/data/nuscenes}"
OUT_DIR="${LCF3D_FRUSTUM_ROOT:-$REPO_ROOT/data/frustum_nuscenes}"

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$REPO_ROOT/src/mmdetection3d:$REPO_ROOT/src:${PYTHONPATH:-}"

for required in \
  "$NUSCENES_ROOT/nuscenes_infos_train.pkl" \
  "$NUSCENES_ROOT/nuscenes_infos_val.pkl" \
  "$NUSCENES_ROOT/v1.0-trainval" \
  "$NUSCENES_ROOT/samples" \
  "$NUSCENES_ROOT/sweeps"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required nuScenes path: $required" >&2
    exit 1
  fi
done

python "$REPO_ROOT/src/dataset_utils/create_frustum_dataset_nuscenes.py" \
  --nuscenes-root "$NUSCENES_ROOT" \
  --out-dir "$OUT_DIR" \
  "$@"
