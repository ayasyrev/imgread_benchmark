import numpy as np
from torchvision import __version__  # noqa: F401
from torchvision.io import read_image


def read_img(img_path: str):
    img = read_image(img_path)
    return img


def read_img_nparray(img_path: str) -> np.ndarray:
    """
    Returns:
        np.ndarray: Image converted to array with shape (width, height, channels)
    """
    image = read_img(img_path)
    return np.transpose(image.numpy(), (1, 2, 0))
