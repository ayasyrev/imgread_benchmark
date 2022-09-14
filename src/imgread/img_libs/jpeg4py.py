import numpy as np
import pkg_resources

import jpeg4py
from PIL import Image


__version__ = pkg_resources.get_distribution("jpeg4py").version


def read_img(img_path: str) -> np.ndarray:
    return jpeg4py.JPEG(img_path).decode()


def read_img_pil(img_path: str) -> Image.Image:
    return Image.fromarray(read_img(img_path))


read_img_nparray = read_img
