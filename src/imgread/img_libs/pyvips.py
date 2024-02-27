import numpy as np
import pyvips
from PIL import Image

__version__ = pyvips.__version__

__all__ = ["read_img", "read_img_ndarray", "read_img_pil", "__version__"]


def read_img(img_path: str) -> np.ndarray:
    """Read image from path with pyvips and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    return pyvips.Image.new_from_file(img_path, access="sequential", memory=True)  # type: ignore


def read_img_ndarray(img_path: str) -> np.ndarray:
    """Reads image from path with pyvips and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    return np.asarray(read_img(img_path))


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with pyvips and returns PIL.Image.

    Returns:
        PIL.Image: Image as PIL.Image
    """
    return Image.fromarray(read_img_ndarray(img_path))
