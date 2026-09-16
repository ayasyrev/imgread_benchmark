"""Adapter for the current imgread package (distinct from legacy imgread_rs)."""

import imgread
from PIL import Image


def read_img_ndarray(path):
    return imgread.load_numpy(path)


def read_img_pil(path):
    return Image.fromarray(read_img_ndarray(path))


read_img = read_img_ndarray


if callable(getattr(imgread, "load_numpy_from_bytes", None)):

    def decode_img_ndarray(data):
        return imgread.load_numpy_from_bytes(data)

    def decode_img_pil(data):
        return Image.fromarray(decode_img_ndarray(data))

    decode_img = decode_img_ndarray
