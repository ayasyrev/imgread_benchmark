import simplejpeg
import numpy as np
from PIL import Image

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img(img_path: str) -> np.ndarray:
    """Read image from path with simplejpeg and returns numpy array."""
    with open(img_path, "rb") as f:
        data = f.read()
    return simplejpeg.decode_jpeg(data, colorspace="RGB")


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with simplejpeg and returns PIL.Image."""
    return Image.fromarray(read_img(img_path))


read_img_ndarray = read_img
