#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LCF3D_CONFIG="$REPO_ROOT/src/model_configs/nuscenes/centerpoint_voxel01_lcf3d_nuscenes_clean.py"
export LCF3D_WORK_DIR="$REPO_ROOT/work_dirs/centerpoint_voxel01_lcf3d_nuscenes_clean"
export LCF3D_AUTO_SCALE_LR="${LCF3D_AUTO_SCALE_LR:-0}"
exec "$REPO_ROOT/src/train_nuscenes.sh" "$@"
