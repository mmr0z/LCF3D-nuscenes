import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mmengine
import numpy as np
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = [
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
]

CAMERA_TYPES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def _to_3x4_intrinsic(cam2img: Iterable[Iterable[float]]) -> np.ndarray:
    cam2img = np.asarray(cam2img, dtype=np.float32)
    if cam2img.shape == (3, 3):
        return np.insert(cam2img, 3, 0.0, axis=1)
    if cam2img.shape == (3, 4):
        return cam2img
    raise ValueError(f'Expected cam2img shape (3, 3) or (3, 4), got {cam2img.shape}')


def _normalize_yaw(yaw: np.ndarray) -> np.ndarray:
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def _random_shift_enlarge_box2d(
        box2d: np.ndarray,
        shift_ratio: float = 0.05,
        enlarge_ratio: float = 0.0) -> np.ndarray:
    xmin, ymin, xmax, ymax = box2d
    height = ymax - ymin
    width = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx = cx + width * shift_ratio * (np.random.random() * 2 - 1)
    cy = cy + height * shift_ratio * (np.random.random() * 2 - 1)
    height = height * (1 + np.random.random() * 2 * shift_ratio - shift_ratio + enlarge_ratio)
    width = width * (1 + np.random.random() * 2 * shift_ratio - shift_ratio + enlarge_ratio)
    return np.asarray([cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0],
                      dtype=np.float32)


def _load_multisweep_points(nusc: NuScenes, sample: dict, sweep_num: int) -> np.ndarray:
    pcl, times = LidarPointCloud.from_file_multisweep(
        nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', sweep_num)
    points = pcl.points.astype(np.float32)
    times = times.reshape(1, -1).astype(np.float32)
    return np.concatenate([points[:4, :], times], axis=0).T


def _points_in_single_box(points_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    center = box[:3]
    dims = box[3:6]
    yaw = box[6]

    shifted = points_xyz - center[None, :]
    cosa = np.cos(yaw)
    sina = np.sin(yaw)
    local_x = shifted[:, 0] * cosa + shifted[:, 1] * sina
    local_y = -shifted[:, 0] * sina + shifted[:, 1] * cosa
    local_z = shifted[:, 2]

    return (
        (np.abs(local_x) <= dims[0] / 2.0) &
        (np.abs(local_y) <= dims[1] / 2.0) &
        (np.abs(local_z) <= dims[2] / 2.0)
    )


def _box_corners_lidar(box: np.ndarray) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = box
    x_corners = np.array([dx, dx, -dx, -dx, dx, dx, -dx, -dx], dtype=np.float32) / 2.0
    y_corners = np.array([dy, -dy, -dy, dy, dy, -dy, -dy, dy], dtype=np.float32) / 2.0
    z_corners = np.array([dz, dz, dz, dz, -dz, -dz, -dz, -dz], dtype=np.float32) / 2.0

    cosa = np.cos(yaw)
    sina = np.sin(yaw)
    corners = np.stack([x_corners, y_corners, z_corners], axis=1)
    rot = np.asarray([
        [cosa, -sina, 0.0],
        [sina, cosa, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return corners @ rot.T + np.asarray([x, y, z], dtype=np.float32)


def _project_points(points_xyz: np.ndarray, lidar2cam: np.ndarray,
                    cam2img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (lidar2cam @ points_h.T).T
    img = (cam2img @ cam.T).T
    valid_depth = img[:, 2] > 1e-5
    img_xy = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float32)
    img_xy[valid_depth] = img[valid_depth, :2] / img[valid_depth, 2:3]
    return img_xy, cam[:, 2]


def _project_box_to_image(box: np.ndarray, lidar2cam: np.ndarray,
                          cam2img: np.ndarray, width: int,
                          height: int) -> np.ndarray:
    corners = _box_corners_lidar(box)
    corners_img, depths = _project_points(corners, lidar2cam, cam2img)
    visible = depths > 1e-5
    if not np.any(visible):
        return None

    valid_corners = corners_img[visible]
    x1 = float(np.nanmin(valid_corners[:, 0]))
    y1 = float(np.nanmin(valid_corners[:, 1]))
    x2 = float(np.nanmax(valid_corners[:, 0]))
    y2 = float(np.nanmax(valid_corners[:, 1]))

    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _camera_image_meta(nusc_root: Path, nusc: NuScenes, token: str) -> Tuple[str, int, int]:
    sample_data = nusc.get('sample_data', token)
    image_path = sample_data['filename']
    with Image.open(nusc_root / image_path) as image:
        width, height = image.size
    return image_path, width, height


def _make_split(
        nusc_root: Path,
        out_path: Path,
        infos_path: Path,
        version: str,
        split_name: str,
        sweep_num: int,
        min_points_per_frustum: int,
        train: bool,
        augmentations: int,
        use_gaussian_likelihoods: bool,
        align_frustum: bool,
        cameras: List[str],
        max_points_per_frustum: int,
        max_samples: int = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    infos = mmengine.load(infos_path)['data_list']
    if max_samples is not None:
        infos = infos[:max_samples]

    nusc = NuScenes(version=version, dataroot=str(nusc_root), verbose=True)
    class_to_label: Dict[int, int] = {i: i for i in range(len(CLASS_NAMES))}
    data_list = []

    for info in tqdm(infos, desc=f'Building {split_name} frustums'):
        sample = nusc.get('sample', info['token'])
        points_full = _load_multisweep_points(nusc, sample, sweep_num)
        points_xyz = points_full[:, :3]
        frustum_base_points = points_full[:, [0, 1, 2, 3]]

        instances = [
            inst for inst in info.get('instances', [])
            if inst.get('bbox_3d_isvalid', True) and int(inst['bbox_label_3d']) in class_to_label
        ]
        if len(instances) == 0:
            continue

        boxes = np.asarray([inst['bbox_3d'] for inst in instances], dtype=np.float32)
        labels = np.asarray([int(inst['bbox_label_3d']) for inst in instances], dtype=np.int64)
        object_masks = [_points_in_single_box(points_xyz, box) for box in boxes]

        for camera_name in cameras:
            image_info = info['images'][camera_name]
            image_path, width, height = _camera_image_meta(
                nusc_root, nusc, image_info['sample_data_token'])
            lidar2cam = np.asarray(image_info['lidar2cam'], dtype=np.float32)
            cam2img = _to_3x4_intrinsic(image_info['cam2img'])
            cam_to_lidar = np.linalg.inv(lidar2cam)

            point_img, point_depth = _project_points(points_xyz, lidar2cam, cam2img)
            depth_mask = point_depth > 1e-5

            for object_idx, (box, label) in enumerate(zip(boxes, labels)):
                bbox_2d = _project_box_to_image(box, lidar2cam, cam2img, width, height)
                if bbox_2d is None:
                    continue

                num_aug = augmentations if train else 1
                for aug_idx in range(num_aug):
                    bbox_aug = (_random_shift_enlarge_box2d(bbox_2d) if train else bbox_2d.copy())
                    fov_mask = (
                        depth_mask &
                        (point_img[:, 0] >= bbox_aug[0]) &
                        (point_img[:, 0] <= bbox_aug[2]) &
                        (point_img[:, 1] >= bbox_aug[1]) &
                        (point_img[:, 1] <= bbox_aug[3])
                    )

                    if int(fov_mask.sum()) <= min_points_per_frustum:
                        continue

                    frustum_points = frustum_base_points[fov_mask].copy()
                    frustum_point_img = point_img[fov_mask]
                    point_mask = object_masks[object_idx][fov_mask].astype(np.float32)

                    if (max_points_per_frustum is not None and
                            frustum_points.shape[0] > max_points_per_frustum):
                        if train:
                            selected = np.random.choice(
                                frustum_points.shape[0],
                                max_points_per_frustum,
                                replace=False)
                        else:
                            selected = np.linspace(
                                0,
                                frustum_points.shape[0] - 1,
                                max_points_per_frustum,
                                dtype=np.int64)
                        frustum_points = frustum_points[selected]
                        frustum_point_img = frustum_point_img[selected]
                        point_mask = point_mask[selected]

                    if use_gaussian_likelihoods:
                        box_width = max(float(bbox_2d[2] - bbox_2d[0]), 1.0)
                        box_height = max(float(bbox_2d[3] - bbox_2d[1]), 1.0)
                        box_center_x = float((bbox_2d[0] + bbox_2d[2]) / 2.0)
                        box_center_y = float((bbox_2d[1] + bbox_2d[3]) / 2.0)
                        likelihoods = np.exp(
                            -((frustum_point_img[:, 0] - box_center_x) ** 2 / (2 * box_width ** 2)) -
                            ((frustum_point_img[:, 1] - box_center_y) ** 2 / (2 * box_height ** 2))
                        ).astype(np.float32)
                        frustum_points = np.concatenate(
                            [frustum_points, likelihoods[:, None]], axis=1)

                    yaw_lidar = 0.0
                    rotation_matrix = np.eye(3, dtype=np.float32)
                    if align_frustum:
                        box_center = np.asarray(
                            [(bbox_aug[0] + bbox_aug[2]) / 2.0,
                             (bbox_aug[1] + bbox_aug[3]) / 2.0,
                             1.0],
                            dtype=np.float32)
                        backproj = np.linalg.inv(cam2img[:, :3]) @ box_center
                        backproj = np.concatenate([backproj, np.asarray([0.0], dtype=np.float32)])
                        backproj = cam_to_lidar @ backproj
                        yaw_lidar = float(np.arctan2(backproj[1], backproj[0]))
                        cosa = np.cos(-yaw_lidar)
                        sina = np.sin(-yaw_lidar)
                        rotation_matrix = np.asarray([
                            [cosa, -sina, 0.0],
                            [sina, cosa, 0.0],
                            [0.0, 0.0, 1.0],
                        ], dtype=np.float32)
                        frustum_points[:, :3] = frustum_points[:, :3] @ rotation_matrix.T

                    bbox_frustum = box.copy()[None, :]
                    bbox_frustum[:, :3] = bbox_frustum[:, :3] @ rotation_matrix.T
                    bbox_frustum[:, -1] = _normalize_yaw(bbox_frustum[:, -1] - yaw_lidar)

                    one_hot = np.zeros(len(CLASS_NAMES), dtype=np.float32)
                    one_hot[label] = 1.0

                    object_id = f'{info["token"]}_{camera_name}_{object_idx:03d}_{aug_idx:02d}'
                    data_list.append({
                        'ori_id': info['token'],
                        'camera_name': camera_name,
                        'object_id': object_id,
                        'inner_sample_id': object_id,
                        'img_path': image_path,
                        'points': frustum_points.astype(np.float32),
                        'pts_semantic_mask': point_mask,
                        'lidar_to_cam': lidar2cam.astype(np.float32),
                        'cam_to_lidar': cam_to_lidar.astype(np.float32),
                        'cam_to_img': cam2img.astype(np.float32),
                        'frustum_angle': yaw_lidar,
                        'gt_bboxes': [bbox_2d.astype(np.float32)],
                        'gt_bboxes_left': [bbox_2d.astype(np.float32)],
                        'gt_labels': [int(label)],
                        'gt_bboxes_3d': bbox_frustum.astype(np.float32),
                        'gt_labels_3d': [int(label)],
                        'one_hot_vector': one_hot,
                    })

    output = {
        'metainfo': {
            'dataset_type': 'frustum_dataset',
            'task_name': 'localization',
            'classes': CLASS_NAMES,
        },
        'data_list': data_list,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f'Wrote {len(data_list)} frustum samples to {out_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create LCF3D nuScenes frustum localizer dataset.')
    parser.add_argument('--nuscenes-root', default='~/mmdetection3d/data/nuscenes')
    parser.add_argument('--out-dir', default='data/frustum_nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--sweep-num', type=int, default=10)
    parser.add_argument('--min-points-per-frustum', type=int, default=10)
    parser.add_argument('--train-augmentations', type=int, default=1)
    parser.add_argument(
        '--max-points-per-frustum',
        type=int,
        default=384,
        help='Cap stored points per frustum to control RAM and pickle size. '
             'Use 0 to disable the cap.')
    parser.add_argument('--no-gaussian-likelihoods', action='store_true')
    parser.add_argument('--no-align-frustum', action='store_true')
    parser.add_argument('--cameras', default=','.join(CAMERA_TYPES))
    parser.add_argument('--max-samples', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_points_per_frustum <= 0:
        args.max_points_per_frustum = None
    nusc_root = Path(args.nuscenes_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    cameras = [camera.strip() for camera in args.cameras.split(',') if camera.strip()]

    for camera in cameras:
        if camera not in CAMERA_TYPES:
            raise ValueError(f'Unknown camera {camera}. Valid cameras: {CAMERA_TYPES}')

    split_specs = [
        ('train', True, args.train_augmentations),
        ('val', False, 1),
    ]
    for split_name, train, augmentations in split_specs:
        infos_path = nusc_root / f'nuscenes_infos_{split_name}.pkl'
        if not infos_path.exists():
            raise FileNotFoundError(f'Missing nuScenes info file: {infos_path}')
        _make_split(
            nusc_root=nusc_root,
            out_path=out_dir / f'nusc_frustum_info_{split_name}.pkl',
            infos_path=infos_path,
            version=args.version,
            split_name=split_name,
            sweep_num=args.sweep_num,
            min_points_per_frustum=args.min_points_per_frustum,
            train=train,
            augmentations=augmentations,
            use_gaussian_likelihoods=not args.no_gaussian_likelihoods,
            align_frustum=not args.no_align_frustum,
            cameras=cameras,
            max_points_per_frustum=args.max_points_per_frustum,
            max_samples=args.max_samples)


if __name__ == '__main__':
    main()
