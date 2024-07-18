import numpy as np
from PIL import Image

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img(img_path: str) -> Image.Image:
    """Read image from path with PIL and returns PIL.Image.

    Returns:
        PIL.Image: Image as PIL.Image
    """
    with open(img_path, "rb") as file:
        img = Image.open(file)
        return img.convert("RGB")


# read_img returns PIL.Image
read_img_pil = read_img


def read_img_ndarray(img_path: str) -> np.ndarray:
    """Reads image from path with PIL and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    return np.asarray(read_img(img_path))
