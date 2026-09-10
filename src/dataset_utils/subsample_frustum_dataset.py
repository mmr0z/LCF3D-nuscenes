#!/usr/bin/env python3
"""Create a class-balanced compact copy of an LCF3D frustum dataset."""

import argparse
import gc
import pickle
import random
from collections import defaultdict
from pathlib import Path


def compact_split(source: Path, destination: Path, cap_per_class: int,
                  seed: int) -> None:
    with source.open('rb') as file:
        dataset = pickle.load(file)

    grouped = defaultdict(list)
    for item in dataset['data_list']:
        label = int(item['gt_labels_3d'][0])
        grouped[label].append(item)

    rng = random.Random(seed)
    selected = []
    for label in sorted(grouped):
        items = grouped[label]
        if len(items) > cap_per_class:
            items = rng.sample(items, cap_per_class)
        selected.extend(items)
    rng.shuffle(selected)

    dataset['data_list'] = selected
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)

    counts = defaultdict(int)
    for item in selected:
        counts[int(item['gt_labels_3d'][0])] += 1
    print(f'{source.name}: {len(selected)} samples, classes={dict(counts)}')

    del dataset, grouped, selected
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-cap-per-class', type=int, default=10000)
    parser.add_argument('--val-cap-per-class', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compact_split(
        args.source_dir / 'nusc_frustum_info_train.pkl',
        args.output_dir / 'nusc_frustum_info_train.pkl',
        args.train_cap_per_class,
        args.seed,
    )
    compact_split(
        args.source_dir / 'nusc_frustum_info_val.pkl',
        args.output_dir / 'nusc_frustum_info_val.pkl',
        args.val_cap_per_class,
        args.seed,
    )


if __name__ == '__main__':
    main()
