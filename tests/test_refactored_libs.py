import numpy as np
import pytest
from PIL import Image
from imgread_benchmark.img_libs import PIL, cv2, skimage, imageio

dog = "tests/test_imgs/dog.jpg"

@pytest.mark.parametrize("lib", [PIL, cv2, skimage, imageio])
def test_standard_interface_pil(lib):
    img = lib.read_img_pil(dog)
    assert isinstance(img, Image.Image)

@pytest.mark.parametrize("lib", [PIL, cv2, skimage, imageio])
def test_standard_interface_ndarray(lib):
    img = lib.read_img_ndarray(dog)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3
    assert img.shape[2] == 3  # RGB
