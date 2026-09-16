import cv2
import numpy as np
from PIL import Image

__all__ = ["read_img", "read_img_ndarray", "read_img_pil"]


def read_img(img_path: str) -> np.ndarray:
    """Read image from path with cv2 and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    # img = cv2.imread(img_path)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)


def read_img_pil(img_path: str) -> Image.Image:
    """Read image from path with cv2 and returns PIL.Image.

    Returns:
        Image.Image: Image as PIL.Image
    """
    return Image.fromarray(read_img(img_path))


# cv2 read image as numpy array
read_img_ndarray = read_img


def decode_img(data):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise ValueError("cv2.imdecode returned None")
    return image


decode_img_ndarray = decode_img


def decode_img_pil(data):
    return Image.fromarray(decode_img(data))
