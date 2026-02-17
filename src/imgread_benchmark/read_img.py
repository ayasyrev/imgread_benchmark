# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
from __future__ import annotations

import importlib
import collections.abc
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Any, Callable, Dict

from numpy import ndarray
from PIL import Image
from rich.console import Console

console = Console()


def _img_libs_pkgs():
    return importlib.import_module("imgread_benchmark.img_libs.img_libs_pkgs")


class _GracefulReadCallable:
    """Pickle-safe callable wrapper with graceful error handling."""

    def __init__(self, func: Callable[..., Any]):
        self._func = func
        self.__wrapped__ = func
        self.__name__ = getattr(func, "__name__", self.__class__.__name__)
        self.__qualname__ = getattr(func, "__qualname__", self.__name__)
        self.__module__ = getattr(func, "__module__", __name__)
        self.__doc__ = getattr(func, "__doc__")

    def __call__(self, img_path: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._func(img_path, *args, **kwargs)
        except (OSError, ValueError) as e:
            console.print(
                f"[yellow]Warning: Error reading {img_path} with "
                f"{self._func.__module__}: {e}[/yellow]"
            )
            return None


def graceful_degradation(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return pickle-safe wrapper for graceful degradation."""
    return _GracefulReadCallable(func)


def load_lib(
    lib_name: str = "pil", module_name: str = "imgread_benchmark.img_libs"
) -> Any:
    """Load image lib."""
    if module_name == "imgread_benchmark.img_libs":
        return _img_libs_pkgs().load_img_lib_adapter(lib_name)
    return importlib.import_module(f"{module_name}.{lib_name}")


@lru_cache(maxsize=1)
def get_img_libs() -> dict:
    """Get loaded image libraries (lazy, cached)."""
    loaded: dict[str, Any] = {}
    for lib in _img_libs_pkgs().get_img_lib_available():
        try:
            loaded[lib] = load_lib(lib)
        except Exception:
            continue
    return loaded


def get_func_dict(
    func_name: str, func_dict: dict[str, Any]
) -> dict[str, Callable[[str], Any]]:
    """Return dict lib_name: func for given func_name"""
    return {
        lib_name: graceful_degradation(func)
        for lib_name, adapter in func_dict.items()
        if (func := getattr(adapter, func_name, None)) is not None
    }


@lru_cache(maxsize=1)
def get_read_img() -> Dict[str, Callable[[str], Any]]:
    """Get read_img function dict (lazy, cached)."""
    return get_func_dict("read_img", get_img_libs())


@lru_cache(maxsize=1)
def get_read_img_pil() -> Dict[str, Callable[[str], Image.Image]]:
    """Get read_img_pil function dict (lazy, cached)."""
    return get_func_dict("read_img_pil", get_img_libs())


@lru_cache(maxsize=1)
def get_read_img_ndarray() -> Dict[str, Callable[[str], ndarray]]:
    """Get read_img_ndarray function dict (lazy, cached)."""
    return get_func_dict("read_img_ndarray", get_img_libs())


@lru_cache(maxsize=1)
def get_read_img_version() -> Dict[str, str]:
    """Get version dict for image libraries (lazy, cached)."""
    versions: Dict[str, str] = {}
    module = _img_libs_pkgs()
    package_map = module.get_lib_package_map()
    for lib_name in module.get_img_lib_available():
        package_name = package_map.get(lib_name)
        if not package_name:
            versions[lib_name] = "unknown"
            continue
        try:
            versions[lib_name] = pkg_version(package_name)
        except PackageNotFoundError:
            versions[lib_name] = "unknown"
    return versions


class _LazyMapping(collections.abc.Mapping):
    def __init__(self, factory: Callable[[], Dict[str, Any]]):
        self._factory = factory

    def _get(self) -> Dict[str, Any]:
        return self._factory()

    def __getitem__(self, key: str) -> Any:
        return self._get()[key]

    def __iter__(self):
        return iter(self._get())

    def __len__(self) -> int:
        return len(self._get())

    def __repr__(self) -> str:
        return repr(self._get())


# For backwards compatibility (lazy)
img_libs = _LazyMapping(get_img_libs)
read_img = _LazyMapping(get_read_img)
read_img_pil = _LazyMapping(get_read_img_pil)
read_img_ndarray = _LazyMapping(get_read_img_ndarray)
read_img_version = _LazyMapping(get_read_img_version)
