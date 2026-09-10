#!/usr/bin/env python3
"""Evaluate an existing nuScenes detection-results JSON without inference."""

import argparse
import json
from pathlib import Path

from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.nuscenes import NuScenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--nuscenes-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--eval-set', default='val')
    parser.add_argument('--eval-version', default='detection_cvpr_2019')
    parser.add_argument('--runtime-json', type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(
        version=args.version, dataroot=str(args.nuscenes_root), verbose=True)
    evaluator = NuScenesEval(
        nusc,
        config=config_factory(args.eval_version),
        result_path=str(args.results),
        eval_set=args.eval_set,
        output_dir=str(args.output_dir),
        verbose=False,
    )
    metrics = evaluator.main(render_curves=False)
    if args.runtime_json:
        metrics['lcf3d_runtime'] = json.loads(args.runtime_json.read_text())
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=4))
    print(f"mAP={metrics['mean_ap']:.6f} NDS={metrics['nd_score']:.6f}")


if __name__ == '__main__':
    main()
