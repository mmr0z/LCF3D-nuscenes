#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FRUSTUM_ROOT="${LCF3D_FRUSTUM_ROOT:-$ROOT/data/frustum_nuscenes_compact}"
CHECKPOINT="${LCF3D_FRUSTUM_CHECKPOINT:-$ROOT/checkpoints/frustum.pth}"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$ROOT/src/mmdetection3d:$ROOT/src:${PYTHONPATH:-}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -f "$FRUSTUM_ROOT/nusc_frustum_info_val.pkl" ]] || { echo "Missing validation data in $FRUSTUM_ROOT" >&2; exit 1; }
python "$ROOT/src/mmdetection3d/tools/test.py"   "$ROOT/src/model_configs/nuscenes/frustum_pointnet_lcf3d_nuscenes.py" "$CHECKPOINT"   --work-dir "$ROOT/work_dirs/frustum_validation"   --cfg-options "test_dataloader.dataset.ann_file=$FRUSTUM_ROOT/nusc_frustum_info_val.pkl" "$@"
