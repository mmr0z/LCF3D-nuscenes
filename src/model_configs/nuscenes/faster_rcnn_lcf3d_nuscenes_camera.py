import os

_base_ = ['./faster_rcnn_lcf3d_nuscenes.py']

nusc_root = os.getenv('LCF3D_NUSCENES_ROOT',
                      os.path.expanduser('~/mmdetection3d/data/nuscenes')).rstrip('/') + '/'
coco_root = os.getenv('LCF3D_CAMERA_COCO_ROOT',
                      './data/nuscenes_camera_coco').rstrip('/') + '/'

work_dir = './work_dirs/faster_rcnn_lcf3d_nuscenes_camera'

metainfo = dict(classes=[
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
])

train_pipeline = [
    dict(
        backend_args=None,
        imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        keep_ratio=True,
        scales=[
            (960, 540),
            (1280, 720),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(
        backend_args=None,
        imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(1280, 720), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file=coco_root + 'nuscenes_camera_train.json',
        backend_args=None,
        data_prefix=dict(img=nusc_root),
        data_root=nusc_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        pipeline=train_pipeline,
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    dataset=dict(
        ann_file=coco_root + 'nuscenes_camera_val.json',
        backend_args=None,
        data_prefix=dict(img=nusc_root),
        data_root=nusc_root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    ann_file=coco_root + 'nuscenes_camera_val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
        'mAP_s',
        'mAP_m',
        'mAP_l',
        'AR@100',
        'AR_s@1000',
        'AR_m@1000',
        'AR_l@1000',
    ],
    type='CocoMetric')

test_evaluator = val_evaluator

train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        save_last=True,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'))
load_from = None
resume = False
