"""Spawn-safe diagnostic datasets and deterministic image fixtures."""

from pathlib import Path

import numpy as np
from PIL import Image


def make_images(root, count=10, shape=(32, 48)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = root / f"{index:03}.png"
        Image.fromarray(np.full((*shape, 3), index, dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


class IndexDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        assert target == 0
        return int(image[0, 0, 0])


def independent_orders(root, reader):
    import torch
    from torch.utils.data import DataLoader
    from imgread_benchmark.dataloader import (
        EpochSampler,
        ImageListDataset,
        snapshot_files,
    )
    from imgread_benchmark.dataloader.engine import close_loader

    manifest = snapshot_files([Path(root) / f"{i:03}.png" for i in range(10)])
    dataset = IndexDataset(ImageListDataset(manifest, reader, False))
    sampler = EpochSampler(10, 19, True)
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    orders = []
    try:
        for epoch in range(3):
            sampler.set_epoch(epoch)
            orders.append([int(i) for batch in loader for i in batch])
            assert (
                orders[-1]
                == torch.randperm(
                    10, generator=torch.Generator().manual_seed(19 + epoch)
                ).tolist()
            )
    finally:
        close_loader(loader, loader._iterator)
    return orders


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(independent_orders(sys.argv[1], sys.argv[2])))
