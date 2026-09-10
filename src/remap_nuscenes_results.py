#!/usr/bin/env python3
"""Remap nuScenes prediction class names and recompute their attributes."""

import argparse
import json
import math
from pathlib import Path


DEFAULT_ATTRIBUTE = {
    'car': 'vehicle.parked',
    'pedestrian': 'pedestrian.moving',
    'trailer': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.moving',
    'motorcycle': 'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked',
    'bicycle': 'cycle.without_rider',
    'barrier': '',
    'traffic_cone': '',
}


def _attribute(name: str, velocity) -> str:
    speed = math.hypot(float(velocity[0]), float(velocity[1]))
    if speed > 0.2:
        if name in {'car', 'construction_vehicle', 'bus', 'truck', 'trailer'}:
            return 'vehicle.moving'
        if name in {'bicycle', 'motorcycle'}:
            return 'cycle.with_rider'
    elif name == 'pedestrian':
        return 'pedestrian.standing'
    elif name == 'bus':
        return 'vehicle.stopped'
    return DEFAULT_ATTRIBUTE[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument(
        '--mapping-json', required=True,
        help='JSON object mapping existing detection_name values to new values')
    args = parser.parse_args()

    mapping = json.loads(args.mapping_json)
    if not isinstance(mapping, dict):
        raise ValueError('--mapping-json must contain a JSON object')

    payload = json.loads(args.input.read_text())
    changed = 0
    for detections in payload['results'].values():
        for detection in detections:
            old_name = detection['detection_name']
            new_name = mapping.get(old_name, old_name)
            if new_name != old_name:
                changed += 1
            detection['detection_name'] = new_name
            detection['attribute_name'] = _attribute(
                new_name, detection.get('velocity', [0.0, 0.0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(f'Wrote {args.output} with {changed} remapped detections.')


if __name__ == '__main__':
    main()
