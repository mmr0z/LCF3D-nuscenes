#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LCF3D_CAMERA_CHECKPOINT="${LCF3D_CAMERA_CHECKPOINT:-$ROOT/checkpoints/camera.pth}"
exec bash "$ROOT/src/test_nuscenes_camera2d.sh" "$@"
