"""CPU image loading benchmark; heavy dependencies are loaded only on use."""

from .manifest import discover_manifest, load_manifest, snapshot_files
from .models import (
    BenchmarkResult,
    BenchmarkRunError,
    DataLoaderConfig,
    FileManifest,
    ReaderInfo,
)

_LAZY = {
    "ImageListDataset": "dataset",
    "ImageTransform": "dataset",
    "EpochSampler": "dataset",
    "run_benchmark": "runner",
    "list_readers": "readers",
    "write_result": "report",
    "read_result": "report",
    "render_result": "report",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    globals()[name] = value
    return value


__all__ = [
    "BenchmarkResult",
    "BenchmarkRunError",
    "DataLoaderConfig",
    "FileManifest",
    "ReaderInfo",
    "discover_manifest",
    "load_manifest",
    "snapshot_files",
    *_LAZY,
]
