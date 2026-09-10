#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="$REPO_ROOT/src/model_configs/nuscenes/centerpoint_voxel01_lcf3d_nuscenes.py"
DEFAULT_WORK_DIR="$REPO_ROOT/work_dirs/centerpoint_voxel01_lcf3d_nuscenes"
CONFIG="${LCF3D_CONFIG:-$DEFAULT_CONFIG}"
WORK_DIR="${LCF3D_WORK_DIR:-$DEFAULT_WORK_DIR}"

export LCF3D_NUSCENES_ROOT="${LCF3D_NUSCENES_ROOT:-$HOME/mmdetection3d/data/nuscenes}"
NUSCENES_ROOT="${LCF3D_NUSCENES_ROOT%/}/"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lcf3d-matplotlib-${USER:-user}}"
export PYTHONPATH="$REPO_ROOT/src/mmdetection3d:$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

required_paths=(
  "${NUSCENES_ROOT}nuscenes_infos_train.pkl"
  "${NUSCENES_ROOT}nuscenes_infos_val.pkl"
  "${NUSCENES_ROOT}nuscenes_dbinfos_train.pkl"
  "${NUSCENES_ROOT}samples"
  "${NUSCENES_ROOT}sweeps"
  "${NUSCENES_ROOT}v1.0-trainval"
)

for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required nuScenes path: $required_path" >&2
    exit 1
  fi
done

# CenterPoint assigns global labels by flattening ``pts_bbox_head.tasks``.
# Both that order and ``class_names`` must match the numeric category ids
# stored in the generated nuScenes info files.
python - "$CONFIG" "${NUSCENES_ROOT}nuscenes_infos_train.pkl" <<'PY'
import pickle
import sys

from mmengine.config import Config

config_path, info_path = sys.argv[1:]
cfg = Config.fromfile(config_path)
with open(info_path, 'rb') as f:
    info = pickle.load(f)

categories = info.get('metainfo', {}).get('categories', {})
annotation_order = [
    name for name, _ in sorted(categories.items(), key=lambda item: item[1])
]
config_order = list(cfg.class_names)
task_order = [
    name
    for task in cfg.model.pts_bbox_head.tasks
    for name in task.class_names
]

if not annotation_order:
    raise SystemExit(f'No metainfo.categories found in {info_path}')
if config_order != annotation_order or task_order != annotation_order:
    print('Inconsistent nuScenes class order; refusing to train:', file=sys.stderr)
    print(f'  annotations: {annotation_order}', file=sys.stderr)
    print(f'  class_names: {config_order}', file=sys.stderr)
    print(f'  head tasks:  {task_order}', file=sys.stderr)
    raise SystemExit(1)
print(f'Validated canonical nuScenes class order: {annotation_order}')
PY

if [[ "${LCF3D_ALLOW_CPU:-0}" != "1" ]]; then
  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    torch_cuda = torch.version.cuda or 'none'
    print(
        f"CUDA is not available to PyTorch in this shell "
        f"(torch={torch.__version__}, torch CUDA={torch_cuda}). "
        "`nvidia-smi` and `python -c \"import torch; print(torch.cuda.is_available())\"` "
        "must both work before starting CenterPoint training. On RTX 50xx, "
        "use `conda activate lcf3d-cu128`; the older `lcf3d` cu117 env will "
        "not initialize this GPU. Set LCF3D_ALLOW_CPU=1 only for debugging "
        "config/data loading.",
        file=sys.stderr)
    sys.exit(1)
PY
fi

train_args=("$CONFIG")
if [[ "${LCF3D_AMP:-0}" == "1" ]]; then
  train_args+=("--amp")
fi
if [[ "${LCF3D_AUTO_SCALE_LR:-1}" == "1" ]]; then
  train_args+=("--auto-scale-lr")
fi

python "$REPO_ROOT/src/mmdetection3d/tools/train.py" \
  "${train_args[@]}" \
  --work-dir "$WORK_DIR" \
  --cfg-options \
  "data_root=$NUSCENES_ROOT" \
  "db_sampler.data_root=$NUSCENES_ROOT" \
  "db_sampler.info_path=${NUSCENES_ROOT}nuscenes_dbinfos_train.pkl" \
  "train_dataloader.dataset.dataset.data_root=$NUSCENES_ROOT" \
  "val_dataloader.dataset.data_root=$NUSCENES_ROOT" \
  "test_dataloader.dataset.data_root=$NUSCENES_ROOT" \
  "val_evaluator.data_root=$NUSCENES_ROOT" \
  "val_evaluator.ann_file=${NUSCENES_ROOT}nuscenes_infos_val.pkl" \
  "test_evaluator.data_root=$NUSCENES_ROOT" \
  "test_evaluator.ann_file=${NUSCENES_ROOT}nuscenes_infos_val.pkl" \
  "$@"
