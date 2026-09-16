import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from PIL import Image
from torch.utils.data import DataLoader

from imgread_benchmark.dataloader import (
    EpochSampler,
    ImageListDataset,
    ImageTransform,
    snapshot_files,
)
from imgread_benchmark.dataloader.dataset import ContextDataset, collate_images
from imgread_benchmark.dataloader.readers import get_reader
from tests.dataloader_helpers import IndexDataset, make_images


@pytest.mark.parametrize("kind", ["normal", "readonly", "negative", "strided", "zero"])
def test_array_tensor_boundary(kind):
    source = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    if kind == "readonly":
        source.flags.writeable = False
    elif kind == "negative":
        source = source[:, ::-1]
    elif kind == "strided":
        source = source[::2, ::2]
    elif kind == "zero":
        source = np.broadcast_to(source[:1], source.shape)
    original = source.copy()
    tensor = torch.from_numpy(source.copy()).permute(2, 0, 1).contiguous()
    transform = ImageTransform()
    assert torch.equal(transform(source), transform(tensor))
    assert np.array_equal(source, original)
    ready = ImageTransform(False)(tensor)
    assert ready is tensor


@pytest.mark.parametrize(
    "height,width,top,left",
    [(256, 257, 16, 16), (256, 259, 16, 18), (256, 514, 16, 145)],
)
def test_crop_coordinates(height, width, top, left):
    y, x = np.indices((height, width))
    source = np.stack((x % 256, y % 256, (x + y) % 256), -1).astype("uint8")
    actual = ImageTransform()(source)
    expected = torch.from_numpy(
        source[top : top + 224, left : left + 224].copy()
    ).permute(2, 0, 1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("shape", [(100, 200), (128, 128), (101, 203)])
def test_resize_geometry_independent_ramp(shape):
    h, w = shape
    # Linear ramps admit an independent pixel-center bilinear expectation.
    y, x = np.indices(shape)
    source = np.stack((x, y, np.full_like(x, 128)), -1).astype("uint8")
    oh, ow = (256, int(256 * w / h)) if h <= w else (int(256 * h / w), 256)
    top, left = round((oh - 224) / 2), round((ow - 224) / 2)
    xx = (np.arange(left, left + 224) + 0.5) * w / ow - 0.5
    yy = (np.arange(top, top + 224) + 0.5) * h / oh - 0.5
    actual = ImageTransform()(source).numpy()
    assert np.max(np.abs(actual[0].astype(float) - xx[None, :])) <= 1
    assert np.max(np.abs(actual[1].astype(float) - yy[:, None])) <= 1
    assert np.all(actual[2] == 128)


@pytest.mark.parametrize(
    "reader", ["pil-rgb", "torchvision-rgb", "cv2-rgb", "cv2-bgr-cvtcolor"]
)
@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "LA"])
def test_reader_semantics(tmp_path, reader, mode):
    path = tmp_path / "image.png"
    colors = {"RGB": (0, 128, 255), "L": 128, "RGBA": (0, 128, 255, 0), "LA": (128, 0)}
    Image.new(mode, (32, 40), colors[mode]).save(path)
    image, target = ImageListDataset(snapshot_files([path]), reader, True)[0]
    assert image.shape == (3, 224, 224) and target == 0
    expected = (128, 128, 128) if mode in ("L", "LA") else (0, 128, 255)
    assert image[:, 0, 0].tolist() == list(expected)


@pytest.mark.parametrize("workers,persistent", [(0, False), (2, False), (2, True)])
@pytest.mark.parametrize("shuffle", [False, True])
def test_actual_orders(tmp_path, workers, persistent, shuffle):
    manifest = snapshot_files(make_images(tmp_path))
    dataset = IndexDataset(ImageListDataset(manifest, "pil-rgb", False))
    sampler = EpochSampler(10, 19, shuffle)
    kwargs = dict(multiprocessing_context="spawn") if workers else {}
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=persistent,
        **kwargs,
    )
    try:
        for epoch in range(3):
            sampler.set_epoch(epoch)
            actual = [int(i) for batch in loader for i in batch]
            expected = (
                torch.randperm(
                    10, generator=torch.Generator().manual_seed(19 + epoch)
                ).tolist()
                if shuffle
                else list(range(10))
            )
            assert actual == expected
    finally:
        from imgread_benchmark.dataloader.engine import close_loader

        close_loader(loader, getattr(loader, "_iterator", None))


def test_native_collation_and_stat(tmp_path):
    paths = make_images(tmp_path, 2)
    Image.new("RGB", (11, 17)).save(paths[1])
    manifest = snapshot_files(paths)
    dataset = ContextDataset(ImageListDataset(manifest, "pil-rgb", False))
    with pytest.raises(ValueError, match="pil-rgb.*shapes"):
        collate_images([dataset[0], dataset[1]])
    assert len(list(DataLoader(dataset, batch_size=1, collate_fn=collate_images))) == 2
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="stat identity"):
        dataset[0]


def test_opencv_no_fallback(monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, "imread", lambda *args: None)
    monkeypatch.setattr(
        cv2, "cvtColor", lambda *args: pytest.fail("cvtColor called on None")
    )
    with pytest.raises(ValueError, match="None"):
        get_reader("cv2-bgr-cvtcolor")("missing")


@pytest.mark.parametrize(
    "reader", ["pil-rgb", "torchvision-rgb", "cv2-rgb", "cv2-bgr-cvtcolor"]
)
def test_exif_orientation_ignored(tmp_path, reader):
    path = tmp_path / "orientation.jpg"
    image = Image.new("RGB", (31, 17), (0, 128, 255))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    output, _ = ImageListDataset(snapshot_files([path]), reader, False)[0]
    assert output.shape == (3, 17, 31)
    assert abs(int(output[1, 0, 0]) - 128) <= 2


def test_antialias_independent_impulse():
    # Downsample by two. The center pixel's triangle-filter weight is 3/8 per axis.
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[256, 256] = 255
    actual = ImageTransform()(image)
    assert abs(int(actual[0, 112, 112]) - round(255 * (3 / 8) ** 2)) <= 1
    assert abs(int(actual[0, 111, 111]) - round(255 * (1 / 8) ** 2)) <= 1


def test_missing_direct_rgb_flag(monkeypatch):
    import cv2
    from imgread_benchmark.dataloader.readers import check_runtime

    monkeypatch.delattr(cv2, "IMREAD_COLOR_RGB")
    with pytest.raises(ValueError, match="IMREAD_COLOR_RGB"):
        check_runtime("cv2-rgb")
