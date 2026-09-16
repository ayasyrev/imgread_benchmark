import ajpegli
import numpy as np
from PIL import Image

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img(img_path: str) -> np.ndarray:
    """Read image from path with ajpegli and returns numpy array."""
    return ajpegli.imread(img_path, mode="RGB")


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with ajpegli and returns PIL.Image."""
    return Image.fromarray(read_img(img_path))


read_img_ndarray = read_img


if callable(getattr(ajpegli, "imdecode", None)):

    def decode_img(data):
        return ajpegli.imdecode(data, mode="RGB")

    decode_img_ndarray = decode_img

    def decode_img_pil(data):
        return Image.fromarray(decode_img(data))
