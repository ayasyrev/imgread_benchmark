# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
import importlib
from typing import Any, Callable, Dict

from numpy import ndarray
from PIL import Image

from .img_libs import img_lib_available


def load_lib(lib_name: str = "pil", module_name: str = "imgread.img_libs") -> Any:
    """Load image lib."""
    return importlib.import_module(f"{module_name}.{lib_name}")


img_libs = {}


for img_lib in img_lib_available:
    img_libs[img_lib] = load_lib(img_lib)


def get_func_dict(func_name: str, img_libs=img_libs):
    return {
        # lib_name: func
        lib_name: getattr(img_libs[lib_name], func_name, None)  # for colab: python 3.7
        for lib_name in img_lib_available
        # if (func := getattr(img_libs[lib_name], func_name, None)) is not None
        if getattr(img_libs[lib_name], func_name, None) is not None
    }


read_img: Dict[str, Callable[[str], Any]] = get_func_dict("read_img")
read_img_pil: Dict[str, Callable[[str], Image.Image]] = get_func_dict("read_img_pil")
read_img_nparray: Dict[str, Callable[[str], ndarray]] = get_func_dict("read_img_nparray")
read_img_version: Dict[str, str] = get_func_dict("__version__")
