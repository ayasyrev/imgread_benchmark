"""One RGB uint8 boundary and one spatial transform for every reader."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

from .manifest import stat_identity
from .models import ImageReadError, validate_storage
from .readers import get_reader


class ImageTransform:
    def __init__(self, geometry=True):
        self.geometry = geometry

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("expected RGB HWC uint8 ndarray")
            if (
                not image.flags.writeable
                or not image.flags.aligned
                or any(s <= 0 for s in image.strides)
            ):
                image = np.array(image, copy=True, order="C")
            image = torch.from_numpy(image).permute(2, 0, 1)
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or image.shape[0] != 3
            or image.dtype != torch.uint8
            or image.device.type != "cpu"
            or min(image.shape) <= 0
        ):
            raise ValueError("expected CPU RGB CHW uint8 Tensor")
        image = image.contiguous()
        if self.geometry:
            image = F.resize(
                image, [256], interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            image = F.center_crop(image, [224, 224])
        return image


class ImageListDataset(Dataset):
    def __init__(self, manifest, reader_id, geometry=True, *, storage="files"):
        validate_storage(reader_id, storage)
        self.manifest = manifest
        self.reader_id = reader_id
        self.storage = storage
        self.transform = ImageTransform(geometry)
        self._reader = None
        self.encoded = None
        if storage == "memory":
            encoded = []
            for entry in manifest.entries:
                try:
                    if stat_identity(entry.path) != entry.stat:
                        raise ValueError("file stat identity changed before preload")
                    data = Path(entry.path).read_bytes()
                    if (
                        stat_identity(entry.path) != entry.stat
                        or len(data) != entry.stat["size"]
                    ):
                        raise ValueError("file stat identity changed during preload")
                    encoded.append(data)
                except Exception as exc:
                    raise ImageReadError(reader_id, entry.path, exc, "preload") from exc
            self.encoded = tuple(encoded)

    def __getstate__(self):
        # Spawned workers construct their own reader, retaining it across samples.
        return {**self.__dict__, "_reader": None}

    @property
    def encoded_bytes(self):
        return sum(map(len, self.encoded)) if self.encoded is not None else 0

    def __len__(self):
        return self.manifest.selected_n

    def __getitem__(self, index):
        entry = self.manifest.entries[index]
        try:
            if self._reader is None:
                self._reader = (
                    get_reader(self.reader_id)
                    if self.storage == "files"
                    else get_reader(self.reader_id, self.storage)
                )
            if self.encoded is None:
                if stat_identity(entry.path) != entry.stat:
                    raise ValueError("file stat identity changed")
                image = self._reader(entry.path)
            else:
                image = self._reader(self.encoded[index])
            return self.transform(image), 0
        except Exception as exc:
            raise ImageReadError(self.reader_id, entry.path, exc) from exc


class ContextDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return (
            image,
            target,
            f"reader={self.dataset.reader_id} path={self.dataset.manifest.entries[index].path}",
        )


def collate_images(samples):
    shapes = [tuple(sample[0].shape) for sample in samples]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"{[s[2] for s in samples]}: incompatible shapes {shapes}; use geometry or batch_size=1"
        )
    return default_collate([(image, target) for image, target, _ in samples])


class EpochSampler:
    def __init__(self, length, seed, shuffle):
        self.length, self.seed, self.shuffle, self.epoch = length, seed, shuffle, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __iter__(self):
        if not self.shuffle:
            return iter(range(self.length))
        generator = torch.Generator(device="cpu").manual_seed(
            (self.seed + self.epoch) % (2**63)
        )
        return iter(torch.randperm(self.length, generator=generator).tolist())
