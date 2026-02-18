import numpy as np
from PIL import Image

import imgread_rs as _imgread_rs

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img_ndarray(img_path: str) -> np.ndarray:
    """Read image with imgread_rs, fallback to PIL until ndarray API is returned."""
    result = _imgread_rs.load_numpy(img_path)
    if result is not None:
        return np.asarray(result)
    with Image.open(img_path) as img:
        return np.asarray(img.convert("RGB"))


def read_img_pil(img_path: str) -> Image.Image:
    """Read image with imgread_rs and return PIL.Image."""
    return Image.fromarray(read_img_ndarray(img_path), mode="RGB")


def read_img(img_path: str) -> np.ndarray:
    """Default reader for imgread_rs backend."""
    return read_img_ndarray(img_path)
