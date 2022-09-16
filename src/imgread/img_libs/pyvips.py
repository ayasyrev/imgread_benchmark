import numpy as np
from PIL import Image

import pyvips

__version__ = pyvips.__version__


def read_img(img_path: str) -> np.ndarray:
    return pyvips.Image.new_from_file(img_path, access="sequential", memory=True)  # type: ignore


def read_img_nparray(img_path: str) -> np.ndarray:
    return np.asarray(read_img(img_path))


def read_img_pil(img_path: str) -> Image.Image:
    return Image.fromarray(read_img_nparray(img_path))
