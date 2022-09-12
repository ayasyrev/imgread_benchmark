import numpy as np
from PIL import Image


def pil2nparray(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


def nparray2pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)
