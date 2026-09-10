_base_ = ['./frustum_pointnet_nuscenes.py']

data_root = './data/frustum_nuscenes/'
work_dir = './work_dirs/frustum_pointnet_lcf3d_nuscenes'

train_dataloader = dict(
    batch_size=64,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        dataset=dict(
            ann_file=data_root + 'nusc_frustum_info_train.pkl')))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        ann_file=data_root + 'nusc_frustum_info_val.pkl'))

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        ann_file=data_root + 'nusc_frustum_info_val.pkl'))

load_from = None
resume = False
