_base_ = ['./faster_rcnn_nuimages_nuscenes.py']

work_dir = './work_dirs/faster_rcnn_lcf3d_nuscenes'

model = dict(
    backbone=dict(init_cfg=None),
    neck=dict(init_cfg=None),
    rpn_head=dict(init_cfg=None))

load_from = None
resume = False
