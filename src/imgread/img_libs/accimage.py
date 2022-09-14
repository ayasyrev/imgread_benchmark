import numpy as np
import pkg_resources

import accimage

__version__ = pkg_resources.get_distribution("accimage").version


def read_img(img_path: str) -> accimage.Image:
    return accimage.Image(img_path)


def read_img_nparray(img_path: str) -> np.ndarray:
    """
    Returns:
        np.ndarray: Image converted to array with shape (width, height, channels)
    """
    image = accimage.Image(img_path)
    image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
    image.copyto(image_np)
    image_np = np.transpose(image_np, (1, 2, 0))
    return image_np


read_img_pil = read_img
