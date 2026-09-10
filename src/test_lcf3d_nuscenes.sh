#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
NUSCENES_ROOT="${LCF3D_NUSCENES_ROOT:-$HOME/mmdetection3d/data/nuscenes}"
FUSION_CONFIG="${LCF3D_FUSION_CFG:-$REPO_ROOT/src/configs/lcf3d_nuscenes_authors_local.json}"
OUTPUT_ROOT="${LCF3D_EVAL_ROOT:-$REPO_ROOT/work_dirs/lcf3d_nuscenes_test}"
DEVICE="${LCF3D_DEVICE:-cuda:0}"
PYTHON_BIN="${LCF3D_PYTHON:-python}"
RUN_NAME="${LCF3D_RUN_NAME:-lcf3d_full_fusion}"
VERSION="${LCF3D_NUSCENES_VERSION:-v1.0-trainval}"
EVAL_SET="${LCF3D_NUSCENES_EVAL_SET:-val}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Activate the LCF3D environment or set LCF3D_PYTHON=/path/to/python." >&2
  exit 1
fi

for required in "$FUSION_CONFIG" "$NUSCENES_ROOT/$VERSION" "$NUSCENES_ROOT/samples"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required LCF3D test path: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$FUSION_CONFIG" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
config = json.loads(config_path.read_text())
missing = []
for section in ('detector2d', 'detector3d'):
    for key in ('cfg_path', 'checkpoint_path'):
        value = config.get(section, {}).get(key)
        if not value or not pathlib.Path(value).expanduser().is_file():
            missing.append(f'{section}.{key}: {value}')
if config.get('use_detection_recovery', True):
    for key in ('cfg_path', 'checkpoint_path'):
        value = config.get('frustum_detector', {}).get(key)
        if not value or not pathlib.Path(value).expanduser().is_file():
            missing.append(f'frustum_detector.{key}: {value}')
if missing:
    print('Fusion test cannot start; missing model files:', file=sys.stderr)
    for item in missing:
        print(f'  - {item}', file=sys.stderr)
    sys.exit(1)
PY

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
cp "$FUSION_CONFIG" "$RUN_DIR/fusion_config.json"

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$REPO_ROOT/src/mmdetection3d:$REPO_ROOT/src:${PYTHONPATH:-}"

"$PYTHON_BIN" "$REPO_ROOT/src/eval_nuscenes.py" \
  -output_dir "$RUN_DIR" \
  -nuscenes_root_path "$NUSCENES_ROOT" \
  -late_fusion_config "$FUSION_CONFIG" \
  -device "$DEVICE" \
  --version "$VERSION" \
  --eval_set "$EVAL_SET" \
  --eval_version detection_cvpr_2019 \
  --sweep_num 10 \
  --use_intensity \
  --cameras CAM_FRONT_LEFT,CAM_FRONT,CAM_FRONT_RIGHT,CAM_BACK_LEFT,CAM_BACK,CAM_BACK_RIGHT \
  "$@"

if [[ -f "$RUN_DIR/metrics.json" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/src/analyze_thesis_metrics.py" \
    "$RUN_DIR" \
    --output-dir "$RUN_DIR/thesis_metrics" \
    --labels "$RUN_NAME"
else
  echo "Official metrics were skipped; raw predictions and runtime are in $RUN_DIR"
fi

echo "LCF3D test and thesis metrics: $RUN_DIR"
