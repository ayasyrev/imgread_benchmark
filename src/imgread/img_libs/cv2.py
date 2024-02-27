import cv2
import numpy as np
from PIL import Image

__version__ = cv2.__version__


def read_img(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_img_pil(img_path: str) -> Image.Image:
    return Image.fromarray(read_img(img_path))


read_img_nparray = read_img
