"""Adapter for the current imgread package (distinct from legacy imgread_rs)."""

import imgread
from PIL import Image


def read_img_ndarray(path):
    return imgread.load_numpy(path)


def read_img_pil(path):
    return Image.fromarray(read_img_ndarray(path))


read_img = read_img_ndarray
