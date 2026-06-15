import imagecodecs
import numpy as np
from PIL import Image

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img(img_path: str) -> np.ndarray:
    """Read image from path with imagecodecs and returns numpy array."""
    img = imagecodecs.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with imagecodecs and returns PIL.Image."""
    return Image.fromarray(read_img(img_path))


read_img_ndarray = read_img
