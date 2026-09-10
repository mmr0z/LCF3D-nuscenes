#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
python -m pip install 'setuptools==60.2.0' wheel ninja
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install --no-build-isolation --no-binary=mmcv 'mmcv==2.1.0'
python -m pip install --no-build-isolation -e ./src/mmdetection3d
python -m pip check
