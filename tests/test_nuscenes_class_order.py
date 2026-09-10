import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CLASSES = [
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


def _python_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f'{name} not found in {path}')


def test_centerpoint_configs_use_nuscenes_annotation_order():
    for name in (
        'centerpoint_voxel01_lcf3d_nuscenes.py',
        'centerpoint_voxel01_lcf3d_nuscenes_clean.py',
    ):
        path = ROOT / 'src/model_configs/nuscenes' / name
        assert _python_assignment(path, 'class_names') == CANONICAL_CLASSES


def test_fusion_configs_do_not_permute_canonical_centerpoint_labels():
    for name in (
        'lcf3d_nuscenes_authors_local.json',
        'lcf3d_nuscenes_two_branch_local.json',
    ):
        config = json.loads((ROOT / 'src/configs' / name).read_text())
        assert config['classes'] == CANONICAL_CLASSES
        assert config['class_mapping_3d'] == list(range(10))
        detector_mapping = config.get('detector3d', {}).get('class_mapping_3d')
        assert detector_mapping in (None, list(range(10)))
