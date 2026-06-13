import ajpegli
import numpy as np
from PIL import Image

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img(img_path: str) -> np.ndarray:
    """Read image from path with ajpegli and returns numpy array."""
    return ajpegli.imread(img_path, mode="RGB")


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with ajpegli and returns PIL.Image."""
    return Image.fromarray(read_img(img_path))


read_img_ndarray = read_img
