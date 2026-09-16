import turbojpeg
import numpy as np
from PIL import Image

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img(img_path: str) -> np.ndarray:
    """Read image from path with turbojpeg and returns numpy array."""
    with open(img_path, "rb") as f:
        data = f.read()
    img = turbojpeg.decompress(data, pixelformat=turbojpeg.PF.RGB)
    return np.array(img)


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with turbojpeg and returns PIL.Image."""
    return Image.fromarray(read_img(img_path))


read_img_ndarray = read_img


def decode_img(data):
    return np.array(turbojpeg.decompress(bytes(data), pixelformat=turbojpeg.PF.RGB))


decode_img_ndarray = decode_img


def decode_img_pil(data):
    return Image.fromarray(decode_img(data))
