#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/checkpoints"
gh release download models-v1 --repo mmr0z/LCF3D-nuscenes --dir "$ROOT/checkpoints" --pattern '*.pth' --skip-existing
cd "$ROOT/checkpoints"
sha256sum --check "$ROOT/docs/models.sha256"
