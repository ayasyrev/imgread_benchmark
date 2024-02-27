import numpy as np
from PIL import Image


def pil2ndarray(img: Image.Image) -> np.ndarray:
    """Convert PIL.Image to numpy array"""
    return np.asarray(img)


def ndarray2pil(img: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL.Image"""
    return Image.fromarray(img)
