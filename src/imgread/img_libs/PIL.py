import numpy as np
from PIL import Image, __version__  # noqa F401


def read_img(img_path: str) -> Image.Image:
    with open(img_path, "rb") as file:
        img = Image.open(file)
        return img.convert("RGB")


read_img_pil = read_img


def read_img_ndarray(img_path: str) -> np.ndarray:
    return np.asarray(read_img(img_path))
