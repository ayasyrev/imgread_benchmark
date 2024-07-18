import accimage
import numpy as np

from importlib.metadata import version
from PIL import Image

__version__ = version("accimage")

__all__ = ["read_img", "read_img_ndarray", "read_img_pil", "__version__"]


def read_img(img_path: str) -> accimage.Image:
    """Read image from path with accimage. Returns accimage.Image."""
    return accimage.Image(img_path)


def read_img_ndarray(img_path: str) -> np.ndarray:
    """
    Reads image from path with accimage and returns numpy array with shape (width, height, channels).
    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    image = accimage.Image(img_path)
    image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
    image.copyto(image_np)
    image_np = np.transpose(image_np, (1, 2, 0))
    return image_np


# accimage by default returns PIL.Image compatible accimage.Image, but sometimes you need PIL.Image
def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with accimage and returns PIL.Image."""
    return Image.fromarray(read_img_ndarray(img_path))
