import imageio
import numpy as np
from PIL import Image

__version__ = imageio.__version__

__all__ = ["read_img", "read_img_ndarray", "read_img_pil", "__version__"]


def read_img(img_path: str) -> np.ndarray:
    """Reads image from path with imageio and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    return imageio.v3.imread(img_path)


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with imageio and returns PIL.Image.

    Returns:
        PIL.Image: Image as PIL.Image
    """
    return Image.fromarray(read_img(img_path))


# read_img returns numpy array
read_img_ndarray = read_img
