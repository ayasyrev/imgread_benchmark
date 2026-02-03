# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
from __future__ import annotations

import importlib
from functools import lru_cache, wraps
from importlib.metadata import version as pkg_version
from typing import Any, Callable, Dict

from numpy import ndarray
from PIL import Image
from rich.console import Console

from .img_libs.img_libs_pkgs import lib_to_package, get_img_lib_available

console = Console()


def graceful_degradation(func: Callable[[str], Any]) -> Callable[[str], Any]:
    """Decorator for graceful degradation."""

    @wraps(func)
    def wrapper(img_path: str, *args, **kwargs) -> Any:
        try:
            return func(img_path, *args, **kwargs)
        except (OSError, ValueError) as e:
            console.print(
                f"[yellow]Warning: Error reading {img_path} with "
                f"{func.__module__}: {e}[/yellow]"
            )
            return None

    return wrapper


def load_lib(
    lib_name: str = "pil", module_name: str = "imgread_benchmark.img_libs"
) -> Any:
    """Load image lib."""
    return importlib.import_module(f"{module_name}.{lib_name}")


@lru_cache(maxsize=1)
def get_img_libs() -> dict:
    """Get loaded image libraries (lazy, cached)."""
    return {lib: load_lib(lib) for lib in get_img_lib_available()}


def get_func_dict(
    func_name: str, func_dict: dict[str, Any]
) -> dict[str, Callable[[str], Any]]:
    """Return dict lib_name: func for given func_name"""
    return {
        lib_name: graceful_degradation(func)
        for lib_name in get_img_lib_available()
        if (func := getattr(func_dict[lib_name], func_name, None)) is not None
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
    return {
        lib_name: pkg_version(lib_to_package[lib_name])
        for lib_name in get_img_lib_available()
    }


class _LazyMapping:
    def __init__(self, factory: Callable[[], Dict[str, Any]]):
        self._factory = factory
        self._value: Dict[str, Any] | None = None

    def _get(self) -> Dict[str, Any]:
        if self._value is None:
            self._value = self._factory()
        return self._value

    def __getitem__(self, key: str) -> Any:
        return self._get()[key]

    def __iter__(self):
        return iter(self._get())

    def __len__(self) -> int:
        return len(self._get())

    def __contains__(self, key: object) -> bool:
        return key in self._get()

    def get(self, key: str, default: Any = None) -> Any:
        return self._get().get(key, default)

    def keys(self):
        return self._get().keys()

    def items(self):
        return self._get().items()

    def values(self):
        return self._get().values()

    def __repr__(self) -> str:
        return repr(self._get())


# For backwards compatibility (lazy)
img_libs = _LazyMapping(get_img_libs)
read_img = _LazyMapping(get_read_img)
read_img_pil = _LazyMapping(get_read_img_pil)
read_img_ndarray = _LazyMapping(get_read_img_ndarray)
read_img_version = _LazyMapping(get_read_img_version)
