# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
import importlib
from typing import Any

from .img_libs import img_lib_available


def load_lib(
    lib_name: str = "pil",
    module_name: str = "imgread.img_libs"
) -> Any:
    """Load image lib."""
    return importlib.import_module(f"{module_name}.{lib_name}")


img_libs = {}


for img_lib in img_lib_available:
    img_libs[img_lib] = load_lib(img_lib)


read_img = {}
read_img_pil = {}
read_img_nparray = {}

if "torchvision" in img_lib_available:
    img_lib_available.pop(-1)
    read_img["torchvision"] = getattr(img_libs["torchvision"], "read_img")

for lib_name in img_lib_available:
    read_img[lib_name] = getattr(img_libs[lib_name], "read_img")
    read_img_pil[lib_name] = getattr(img_libs[lib_name], "read_img_pil")
    read_img_nparray[lib_name] = getattr(img_libs[lib_name], "read_img_nparray")
