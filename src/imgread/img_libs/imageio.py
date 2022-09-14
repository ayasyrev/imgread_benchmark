import numpy as np

import imageio
from PIL import Image

__version__ = imageio.__version__


def read_img(img_path: str) -> np.ndarray:
    return imageio.v2.imread(img_path)


def read_img_pil(img_path: str) -> Image.Image:
    return Image.fromarray(read_img(img_path))


read_img_nparray = read_img
