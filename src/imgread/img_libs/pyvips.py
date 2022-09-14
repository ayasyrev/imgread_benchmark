import numpy as np
from PIL import Image

import pyvips

__version__ = pyvips.__version__


def read_img(img_path: str) -> np.ndarray:
    image = pyvips.Image.new_from_file(img_path, access="sequential", memory=True)

    numpy_image = np.ndarray(
        dtype=np.uint8, shape=[image.height, image.width, image.bands]  # type: ignore
    )

    return numpy_image


def read_img_pil(img_path: str) -> Image.Image:
    return Image.fromarray(read_img(img_path))


read_img_nparray = read_img
