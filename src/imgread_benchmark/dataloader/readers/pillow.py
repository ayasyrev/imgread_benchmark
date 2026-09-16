from io import BytesIO

import numpy as np
from PIL import Image


def read(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_bytes(data):
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))
