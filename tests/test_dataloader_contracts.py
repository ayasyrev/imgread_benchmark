import json

import pytest
from PIL import Image

from imgread_benchmark.dataloader import DataLoaderConfig, FileManifest, snapshot_files
from imgread_benchmark.dataloader.manifest import validate_manifest
from imgread_benchmark.dataloader.models import EpochResult


@pytest.fixture
def files(tmp_path):
    result = []
    for i in range(3):
        path = tmp_path / f"картинка {i}.png"
        Image.new("RGB", (20, 30), (i, 128, 255)).save(path)
        result.append(path)
    return result


def test_snapshot(files):
    original = [files[2], files[0], files[2]]
    manifest = snapshot_files(original)
    original.clear()
    assert manifest.paths == tuple(map(str, [files[2], files[0], files[2]]))
    assert (
        FileManifest.from_dict(json.loads(json.dumps(manifest.to_dict()))) == manifest
    )
    assert snapshot_files(manifest.paths).selection_id == manifest.selection_id
    assert snapshot_files(files, num_samples=2).selected_n == 2
    for source, n in [([], 0), (files, -1)]:
        with pytest.raises(ValueError):
            snapshot_files(source, num_samples=n)
    data = manifest.to_dict()
    data["selection_id"] = "bad"
    with pytest.raises(ValueError):
        FileManifest.from_dict(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_workers": -1},
        {"batch_size": 0},
        {"epochs": 0},
        {"seed": -1},
        {"seed": 2**63},
        {"seed": 1.5},
        {"sample_interval_ms": float("nan")},
        {"sample_interval_ms": 0},
        {"prefetch_factor": 2},
        {"persistent_workers": True},
        {"num_workers": 2, "prefetch_factor": 0},
        {"reader": "fallback"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DataLoaderConfig(**kwargs)


def test_effective_config():
    assert DataLoaderConfig().effective_prefetch_factor is None
    assert DataLoaderConfig(num_workers=2).effective_prefetch_factor == 2


@pytest.mark.parametrize(
    "mode,fmt",
    [
        ("RGB", "BMP"),
        ("RGB", "TIFF"),
        ("CMYK", "JPEG"),
        ("I;16", "PNG"),
        ("P", "PNG"),
        ("1", "PNG"),
    ],
)
def test_source_rejection(tmp_path, mode, fmt):
    path = tmp_path / "misleading.jpg"
    Image.new(mode, (10, 10)).save(path, format=fmt)
    with pytest.raises(ValueError, match=str(path)):
        snapshot_files([path])


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "LA"])
def test_supported_png(tmp_path, mode):
    path = tmp_path / "actually_png.jpg"
    Image.new(mode, (10, 10)).save(path, format="PNG")
    manifest = snapshot_files([path])
    validate_manifest(manifest, "pil-rgb")
    assert manifest.entries[0].header["format"] == "PNG"


def test_rates():
    epoch = EpochResult.measured(0, 0, 2_000_000_000, 10, 3, 0, "a", "b")
    assert epoch.ms_per_image == 200
    assert epoch.images_per_second == 5
    assert (
        EpochResult.measured(
            0, 0, 1, 0, 0, 0, "a", "b", status="failed"
        ).images_per_second
        is None
    )


def test_png_rgb_16bit_and_animation(tmp_path):
    import struct
    import zlib

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    path = tmp_path / "16rgb.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 16, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" * 26))
        + chunk(b"IEND", b"")
    )
    with Image.open(path) as image:
        assert image.mode == "RGB"  # decoded mode alone would hide the source depth
    with pytest.raises(ValueError, match="bit_depth=16"):
        snapshot_files([path])
    animated = tmp_path / "animated.png"
    Image.new("RGB", (4, 4), "red").save(
        animated,
        save_all=True,
        append_images=[Image.new("RGB", (4, 4), "blue")],
        duration=100,
        loop=0,
    )
    with pytest.raises(ValueError, match="frames=2"):
        snapshot_files([animated])


def test_header_validation_does_not_decode(files, monkeypatch):
    from PIL import PngImagePlugin

    monkeypatch.setattr(
        PngImagePlugin.PngImageFile,
        "load",
        lambda *args: pytest.fail("header preflight decoded pixels"),
    )
    snapshot_files(files)


def test_discovery_keeps_unsupported_in_selected_snapshot(tmp_path):
    from imgread_benchmark.dataloader import discover_manifest

    Image.new("RGB", (4, 4)).save(tmp_path / "a.bmp")
    with pytest.raises(ValueError, match="BMP"):
        discover_manifest(tmp_path)


@pytest.mark.parametrize(
    "value", [None, [], {}, {"schema_version": 1}, {"schema_version": 2}]
)
def test_malformed_manifest(value):
    with pytest.raises(ValueError):
        FileManifest.from_dict(value)
