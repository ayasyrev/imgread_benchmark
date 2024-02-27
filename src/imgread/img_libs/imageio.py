import imageio
import numpy as np
from PIL import Image

__version__ = imageio.__version__


def read_img(img_path: str) -> np.ndarray:
    return imageio.imread(img_path)


def read_img_pil(img_path: str) -> Image.Image:
    return Image.fromarray(read_img(img_path))


read_img_ndarray = read_img
