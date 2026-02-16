import numpy as np
from PIL import Image

import local_rs as _local_rs

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img_ndarray(img_path: str) -> np.ndarray:
    """Read image from path with local_rs and return numpy array."""
    try:
        return np.asarray(_local_rs.open_jpeg_turbo(img_path))
    except Exception:
        return np.asarray(_local_rs.open_jpeg(img_path))


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with local_rs and return PIL.Image."""
    return Image.fromarray(read_img_ndarray(img_path), mode="RGB")


def read_img(img_path: str) -> np.ndarray:
    """Default reader for local_rs backend."""
    return read_img_ndarray(img_path)
