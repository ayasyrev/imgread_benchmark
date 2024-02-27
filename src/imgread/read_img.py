# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
import importlib
from typing import Any, Callable, Dict

from numpy import ndarray
from PIL import Image

from .img_libs.img_libs_pkgs import img_lib_available


def load_lib(lib_name: str = "pil", module_name: str = "imgread.img_libs") -> Any:
    """Load image lib."""
    return importlib.import_module(f"{module_name}.{lib_name}")


img_libs = {}


for img_lib in img_lib_available:
    img_libs[img_lib] = load_lib(img_lib)


def get_func_dict(func_name: str, func_dict: dict) -> dict[str, Callable[[str], Any]]:
    """Return dict lib_name: func for given func_name"""
    return {
        lib_name: func
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
read_img_version: Dict[str, str] = get_func_dict("__version__", img_libs)
