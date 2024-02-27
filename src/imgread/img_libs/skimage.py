import numpy as np
import skimage
import skimage.io as io
from PIL import Image

__version__ = skimage.__version__


def read_img(img_path: str) -> np.ndarray:
    return io.imread(img_path, plugin="matplotlib")


def read_img_pil(img_path: str) -> Image.Image:
    return Image.fromarray(read_img(img_path))


read_img_ndarray = read_img
