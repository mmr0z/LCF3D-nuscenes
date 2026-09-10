#!/usr/bin/env python
"""Convert MMDetection3D nuScenes infos with camera boxes to COCO 2D.

The generated annotations use one COCO image per nuScenes camera frame and
2D boxes from the `cam_instances` field produced by the MMDetection3D
nuScenes converter.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

import mmengine


CAMERAS = (
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
)

CLASSES = (
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
)


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _load_infos(info_path: Path) -> list[dict]:
    data = mmengine.load(str(info_path))
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported nuScenes info format in {info_path}')


def convert_split(info_path: Path,
                  nusc_root: Path,
                  out_path: Path,
                  cameras: tuple[str, ...],
                  width: int,
                  height: int,
                  min_area: float,
                  max_samples: int | None = None) -> None:
    infos = _load_infos(info_path)
    if max_samples is not None:
        infos = infos[:max_samples]

    images = []
    annotations = []
    missing_images = 0
    image_id = 1
    ann_id = 1

    for info in infos:
        image_infos = info.get('images', {})
        cam_instances = info.get('cam_instances', {})

        for camera in cameras:
            if camera not in image_infos:
                continue

            image_info = image_infos[camera]
            file_name = f'samples/{camera}/{image_info["img_path"]}'
            image_path = nusc_root / file_name
            if not image_path.exists():
                missing_images += 1
                continue

            images.append({
                'id': image_id,
                'file_name': file_name,
                'width': width,
                'height': height,
                'sample_token': info.get('token', ''),
                'camera': camera,
            })

            for instance in cam_instances.get(camera, []):
                label = int(instance.get('bbox_label', -1))
                if label < 0 or label >= len(CLASSES):
                    continue

                x1, y1, x2, y2 = _as_float_list(instance['bbox'])
                x1 = max(0.0, min(float(width), x1))
                y1 = max(0.0, min(float(height), y1))
                x2 = max(0.0, min(float(width), x2))
                y2 = max(0.0, min(float(height), y2))
                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h
                if box_w <= 0.0 or box_h <= 0.0 or area < min_area:
                    continue

                annotations.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': label + 1,
                    'bbox': [x1, y1, box_w, box_h],
                    'area': area,
                    'iscrowd': 0,
                })
                ann_id += 1

            image_id += 1

    coco = {
        'info': {
            'description': 'nuScenes camera 2D boxes converted from MMDetection3D infos',
            'source_info': str(info_path),
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [
            {
                'id': idx + 1,
                'name': name,
            } for idx, name in enumerate(CLASSES)
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as out_file:
        json.dump(coco, out_file)

    print(
        f'Wrote {out_path}: {len(images)} images, '
        f'{len(annotations)} annotations, {missing_images} missing images skipped')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert nuScenes MMDetection3D info files to COCO 2D camera annotations.')
    parser.add_argument(
        '--nuscenes-root',
        default='~/mmdetection3d/data/nuscenes',
        type=Path,
        help='nuScenes root containing samples/, sweeps/ and nuscenes_infos_*.pkl')
    parser.add_argument(
        '--out-dir',
        default='data/nuscenes_camera_coco',
        type=Path,
        help='Directory where COCO JSON files will be written')
    parser.add_argument('--train-info', default='nuscenes_infos_train.pkl')
    parser.add_argument('--val-info', default='nuscenes_infos_val.pkl')
    parser.add_argument(
        '--cameras',
        default=','.join(CAMERAS),
        help='Comma-separated camera names to export')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--min-area', type=float, default=1.0)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional sample cap per split for debugging')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nusc_root = args.nuscenes_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    cameras = tuple(camera.strip() for camera in args.cameras.split(',') if camera.strip())

    convert_split(
        nusc_root / args.train_info,
        nusc_root,
        out_dir / 'nuscenes_camera_train.json',
        cameras,
        args.width,
        args.height,
        args.min_area,
        args.max_samples,
    )
    convert_split(
        nusc_root / args.val_info,
        nusc_root,
        out_dir / 'nuscenes_camera_val.json',
        cameras,
        args.width,
        args.height,
        args.min_area,
        args.max_samples,
    )


if __name__ == '__main__':
    main()
