# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
from __future__ import annotations

import importlib
import logging
from functools import wraps
from importlib.metadata import version as pkg_version
from typing import Any, Callable, Dict

from numpy import ndarray
from PIL import Image
from rich.console import Console

from .img_libs.img_libs_pkgs import img_lib_available, lib_to_package

console = Console()


def graceful_degradation(func: Callable[[str], Any]) -> Callable[[str], Any]:
    """Decorator for graceful degradation."""

    @wraps(func)
    def wrapper(img_path: str, *args, **kwargs) -> Any:
        try:
            return func(img_path, *args, **kwargs)
        except Exception as e:
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


img_libs = {}


for img_lib in img_lib_available:
    img_libs[img_lib] = load_lib(img_lib)


def get_func_dict(
    func_name: str, func_dict: dict[str, Any]
) -> dict[str, Callable[[str], Any]]:
    """Return dict lib_name: func for given func_name"""
    return {
        lib_name: graceful_degradation(func)
        for lib_name in img_lib_available
        if (func := getattr(func_dict[lib_name], func_name, None)) is not None
    }


read_img: Dict[str, Callable[[str], Any]] = get_func_dict("read_img", img_libs)
read_img_pil: Dict[str, Callable[[str], Image.Image]] = get_func_dict(
    "read_img_pil", img_libs
)
read_img_ndarray: Dict[str, Callable[[str], ndarray]] = get_func_dict(
    "read_img_ndarray", img_libs
)
read_img_version: Dict[str, str] = {
    lib_name: pkg_version(lib_to_package[lib_name]) for lib_name in img_lib_available
}
