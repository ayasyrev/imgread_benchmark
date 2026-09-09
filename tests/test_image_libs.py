import numpy as np
import pytest
from PIL import Image

from imgread_benchmark.img_libs.img_libs_pkgs import img_lib_available
from imgread_benchmark.read_img import read_img, read_img_ndarray, read_img_pil

image_readers = [name for name in img_lib_available if name != "torchvision"]

dog = "tests/test_imgs/dog.jpg"


@pytest.mark.parametrize("img_lib", image_readers)
def test_read_img_pil(img_lib):
    img = read_img_pil[img_lib](dog)
    assert isinstance(img, Image.Image)
    assert img.size == (224, 224)


@pytest.mark.parametrize("img_lib", image_readers)
def test_read_img_ndarray(img_lib):
    img = read_img_ndarray[img_lib](dog)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.shape == (224, 224, 3)


@pytest.mark.parametrize("img_lib", image_readers)
def test_read_img(img_lib):
    img = read_img[img_lib](dog)
    assert img is not None


def test_read_img_torchvision():
    pytest.importorskip("torchvision")
    img = read_img["torchvision"](dog)
    assert img.shape == (3, 224, 224)
    img = read_img_ndarray["torchvision"](dog)
    assert img.shape == (224, 224, 3)
    img = read_img_pil["torchvision"](dog)
    assert img.size == (224, 224)
