import jpeg4py
import numpy as np

from PIL import Image

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img(img_path: str) -> np.ndarray:
    """Read image from path with jpeg4py and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    return jpeg4py.JPEG(img_path).decode()


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with jpeg4py and returns PIL.Image.

    Returns:
        PIL.Image: Image as PIL.Image
    """
    return Image.fromarray(read_img(img_path))


# read_img returns numpy array
read_img_ndarray = read_img
