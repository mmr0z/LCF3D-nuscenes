#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LCF3D_FUSION_CFG="${LCF3D_FUSION_CFG:-$ROOT/src/configs/lcf3d_nuscenes.json}"
exec bash "$ROOT/src/test_lcf3d_nuscenes.sh" "$@"
