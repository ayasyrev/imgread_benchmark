"""Encoded-memory mode and persistent imgread use the existing loading pipeline."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
import torch

from imgread_benchmark.cli import _build_cli
from imgread_benchmark.dataloader import (
    BenchmarkRunError,
    DataLoaderConfig,
    EpochSampler,
    ImageListDataset,
    run_benchmark,
    snapshot_files,
)
from imgread_benchmark.dataloader.engine import close_loader
from imgread_benchmark.dataloader.models import ImageReadError
from imgread_benchmark.dataloader.readers import check_runtime, get_reader
from imgread_benchmark.dataloader.report import read_result
from imgread_benchmark.dataloader.runner import group_members
from tests.dataloader_helpers import IndexDataset, make_images

MEMORY_READERS = (
    "pil-rgb",
    "torchvision-rgb",
    "cv2-rgb",
    "cv2-bgr-cvtcolor",
    "imgread-rgb",
    "imgread-loader-rgb",
)


def require_reader(reader, storage="memory"):
    if reader.startswith("imgread"):
        module = pytest.importorskip("imgread")
        if reader == "imgread-loader-rgb" and not hasattr(module, "Loader"):
            pytest.skip("install an imgread build providing Loader")
        if (
            reader == "imgread-loader-rgb"
            and storage == "memory"
            and not callable(getattr(module.Loader, "decode", None))
        ):
            pytest.skip("install an imgread build providing Loader.decode")


@pytest.mark.parametrize("reader", MEMORY_READERS)
@pytest.mark.parametrize(
    "mode,fmt",
    [
        ("RGB", "PNG"),
        ("L", "PNG"),
        ("RGBA", "PNG"),
        ("LA", "PNG"),
        ("RGB", "JPEG"),
        ("L", "JPEG"),
    ],
)
def test_encoded_memory_matches_files_without_disk_reads(tmp_path, reader, mode, fmt):
    require_reader(reader)
    path = tmp_path / "image.bin"
    y, x = np.indices((19, 31))
    pixels = np.stack((x * 7, y * 11, (x + y) * 5), -1).astype("uint8")
    image = Image.fromarray(pixels).convert(mode)
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, format=fmt, exif=exif)
    manifest = snapshot_files([path, path])
    files = ImageListDataset(manifest, reader, False)
    expected, _ = files[0]
    memory = ImageListDataset(manifest, reader, False, storage="memory")
    assert len(memory) == 2
    assert memory.encoded_bytes == 2 * path.stat().st_size
    path.unlink()
    for index in (0, 1, 0):
        actual, target = memory[index]
        assert target == 0 and actual.shape == (3, 19, 31)
        assert actual.dtype == torch.uint8 and actual.is_contiguous()
        assert torch.equal(actual, expected)
        actual.fill_(0)  # The next access must decode a new output, not reuse pixels.


def test_preload_detects_changed_file(tmp_path):
    paths = make_images(tmp_path, 1)
    manifest = snapshot_files(paths)
    paths[0].write_bytes(b"changed")
    with pytest.raises(ImageReadError, match="changed before preload") as caught:
        ImageListDataset(manifest, "pil-rgb", storage="memory")
    assert caught.value.stage == "preload"
    assert caught.value.path == str(paths[0])


@pytest.mark.parametrize("reader", MEMORY_READERS)
def test_decode_bytes_errors_propagate(reader):
    require_reader(reader)
    with pytest.raises(Exception):
        get_reader(reader, "memory")(b"not an image")


@pytest.mark.parametrize("storage", ["files", "memory"])
def test_persistent_reader_reused(tmp_path, monkeypatch, storage):
    imgread = pytest.importorskip("imgread")
    calls, instances = [], []

    class Reader:
        def __init__(self, **kwargs):
            assert kwargs == dict(color="rgb", dtype="uint8", backend="auto")
            instances.append(self)

        def __call__(self, path):
            assert storage == "files"
            calls.append((self, path))
            return np.zeros((5, 7, 3), dtype=np.uint8)

        def decode(self, data):
            assert storage == "memory" and isinstance(data, bytes)
            calls.append((self, data))
            return np.zeros((5, 7, 3), dtype=np.uint8)

    monkeypatch.setattr(imgread, "Loader", Reader, raising=False)
    manifest = snapshot_files(make_images(tmp_path, 2))
    dataset = ImageListDataset(manifest, "imgread-loader-rgb", False, storage=storage)
    assert not instances  # Construct the Loader lazily in its owning process.
    if storage == "memory":
        for path in manifest.paths:
            Path(path).unlink()
    for index in (0, 1, 0):
        dataset[index]
    assert len(instances) == 1
    assert all(owner is instances[0] for owner, _ in calls)
    inputs = manifest.paths if storage == "files" else dataset.encoded
    assert [value for _, value in calls] == [inputs[index] for index in (0, 1, 0)]


@pytest.mark.parametrize(
    "reader,storage",
    [
        *((reader, "memory") for reader in MEMORY_READERS),
        ("imgread-loader-rgb", "files"),
    ],
)
def test_spawn_persistent_epochs_preserve_pixels_and_order(tmp_path, reader, storage):
    require_reader(reader, storage)
    manifest = snapshot_files(make_images(tmp_path, 5))
    dataset = ImageListDataset(manifest, reader, False, storage=storage)
    dataset[0]  # Also exercise serialization of an already-used reader.
    if storage == "memory":
        for path in manifest.paths:
            Path(path).unlink()
    sampler = EpochSampler(5, 11, True)
    loader = torch.utils.data.DataLoader(
        IndexDataset(dataset),
        sampler=sampler,
        batch_size=2,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    try:
        for epoch in range(2):
            sampler.set_epoch(epoch)
            actual = [int(index) for batch in loader for index in batch]
            assert actual == list(sampler)
    finally:
        close_loader(loader, getattr(loader, "_iterator", None))


@pytest.mark.parametrize(
    "reader,storage,workers",
    [
        ("pil-rgb", "memory", 0),
        ("imgread-rgb", "memory", 2),
        ("imgread-rgb", "files", 0),
        ("imgread-loader-rgb", "files", 2),
        ("imgread-loader-rgb", "memory", 0),
        ("imgread-loader-rgb", "memory", 2),
    ],
)
def test_existing_runner_storage(tmp_path, reader, storage, workers):
    require_reader(reader, storage)
    manifest = snapshot_files(make_images(tmp_path, 5))
    config = DataLoaderConfig(
        reader=reader,
        storage=storage,
        num_workers=workers,
        persistent_workers=bool(workers),
        epochs=2,
        batch_size=2,
        geometry=False,
        shuffle=True,
        monitor_resources=True,
    )
    result = run_benchmark(manifest, config, timeout_seconds=60)
    assert result.status == "success"
    assert result.config["effective"]["storage"] == storage
    assert result.preparation["storage"] == storage
    assert all(
        e.images_delivered == 5 and e.batches_delivered == 3 for e in result.epochs
    )
    if storage == "memory":
        assert result.preparation["encoded_bytes"] == sum(
            e.stat["size"] for e in manifest.entries
        )
        assert result.preparation["warmup"] is False
    assert len(result.preparation["worker_registrations"]) == workers
    assert not group_members(result.consumer["pid"])


@pytest.mark.parametrize("reader", ["pil-rgb", "imgread-loader-rgb"])
def test_memory_corrupt_payload_retains_failure_context(tmp_path, reader):
    import struct

    require_reader(reader)
    path = make_images(tmp_path, 1)[0]
    data = bytearray(path.read_bytes())
    offset = data.index(b"IDAT")
    size = struct.unpack(">I", data[offset - 4 : offset])[0]
    data[offset + 4 : offset + 4 + size] = b"x" * size
    path.write_bytes(data)
    manifest = snapshot_files([path])
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmark(
            manifest,
            DataLoaderConfig(reader=reader, storage="memory", num_workers=2, epochs=1),
        )
    result = caught.value.result
    assert str(path) in result.error["reason"]
    assert result.epochs[0].images_per_second is None
    assert not group_members(result.consumer["pid"])


@pytest.mark.parametrize("reader", ["pil-rgb", "imgread-loader-rgb"])
def test_memory_cli_and_storage_config_identity(tmp_path, reader):
    require_reader(reader)
    paths = make_images(tmp_path / "images", 3)
    output = tmp_path / "memory"
    _build_cli()(
        [
            "dataloader",
            str(paths[0].parent),
            "--reader",
            reader,
            "--storage",
            "memory",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--no-geometry",
            "--output",
            str(output),
        ]
    )
    result = read_result(output)
    assert result.config["requested"]["storage"] == "memory"
    assert result.preparation["encoded_bytes"] > 0
    assert result.epochs[0].images_delivered == 3
    files = run_benchmark(
        result.manifest,
        DataLoaderConfig(reader=reader, epochs=1, batch_size=2, geometry=False),
    )
    assert files.config_id != result.config_id
    assert files.epochs[0].order_id == result.epochs[0].order_id
    assert files.epochs[0].delivered_order_id == result.epochs[0].delivered_order_id


def test_unsupported_storage_and_missing_loader_are_explicit(monkeypatch):
    with pytest.raises(ValueError, match="Unknown storage"):
        DataLoaderConfig(storage="decoded")
    DataLoaderConfig(reader="imgread-loader-rgb", storage="memory")
    imgread = pytest.importorskip("imgread")
    monkeypatch.delattr(imgread, "Loader", raising=False)
    for storage in ("files", "memory"):
        with pytest.raises(ValueError, match="missing required API Loader"):
            check_runtime("imgread-loader-rgb", storage)


@pytest.mark.parametrize("decode", [None, False])
def test_missing_decode_rejected_only_for_memory(monkeypatch, capsys, decode):
    import json
    import sys
    from imgread_benchmark.dataloader.readers import _probe

    imgread = pytest.importorskip("imgread")

    class FileLoader:
        def __init__(self, **kwargs):
            pass

        def __call__(self, path):
            pytest.fail("capability check must not decode images")

    if decode is not None:
        FileLoader.decode = decode
    monkeypatch.setattr(imgread, "Loader", FileLoader, raising=False)
    assert check_runtime("imgread-loader-rgb", "files").available
    for probe in (get_reader, check_runtime):
        with pytest.raises(ValueError, match="missing required API Loader.decode"):
            probe("imgread-loader-rgb", "memory")
    monkeypatch.setattr(sys, "argv", ["-c", "imgread-loader-rgb", "memory"])
    _probe()
    result = json.loads(capsys.readouterr().out)
    assert not result["available"]
    assert "Loader.decode" in result["reason"]


def test_memory_capability_checked_before_consumer_start(tmp_path, monkeypatch):
    from imgread_benchmark.dataloader import readers, runner
    from imgread_benchmark.dataloader.models import ReaderInfo

    def probe(reader, storage):
        assert reader == "imgread-loader-rgb" and storage == "memory"
        return ReaderInfo(reader, "test", False, "missing required API Loader.decode")

    monkeypatch.setattr(readers, "probe_reader", probe)
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("unsupported reader started a consumer"),
    )
    with pytest.raises(ValueError, match="Loader.decode"):
        run_benchmark(
            snapshot_files(make_images(tmp_path, 1)),
            DataLoaderConfig(reader="imgread-loader-rgb", storage="memory"),
        )
