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


_buffer_decoder = getattr(_local_rs, "open_jpeg_turbo_from_bytes", None)
if not callable(_buffer_decoder):
    _buffer_decoder = getattr(_local_rs, "open_jpeg_from_bytes", None)
if callable(_buffer_decoder):

    def decode_img(data):
        result = _buffer_decoder(bytes(data))
        if result is None:
            raise ValueError("local_rs returned None")
        return np.asarray(result)

    decode_img_ndarray = decode_img

    def decode_img_pil(data):
        return Image.fromarray(decode_img(data))
