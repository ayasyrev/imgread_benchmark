from pathlib import Path

import pytest

from imgread.img_libs.img_libs_pkgs import img_lib_available, lib_to_package
from imgread.read_img import read_img, read_img_ndarray, read_img_pil

if "torchvision" in img_lib_available:  # torchvision test separately
    img_lib_available.pop(-1)  # pragma: no cover

dog = 'tests/test_imgs/dog.jpg'


def test_libs_list():
    assert "torchvision" not in img_lib_available
    img_path = Path(dog)
    assert img_path.exists()


@pytest.mark.parametrize("img_lib", img_lib_available)
def test_read_img_pil(img_lib):
    assert img_lib in lib_to_package
    img = read_img_pil[img_lib](dog)
    assert img.size == (224, 224)


@pytest.mark.parametrize("img_lib", img_lib_available)
def test_read_img_ndarray(img_lib):
    assert img_lib in lib_to_package
    img = read_img_ndarray[img_lib](dog)
    assert img.size == 150528
    assert img.shape == (224, 224, 3)


@pytest.mark.parametrize("img_lib", img_lib_available)
def test_read_img(img_lib):
    assert img_lib in lib_to_package
    img = read_img[img_lib](dog)
    assert img is not None


def test_read_img_torchvision():
    assert "torchvision" in lib_to_package
    img = read_img["torchvision"](dog)
    assert img.shape[0] == 3
    assert img.shape[1] == 224
    assert img.shape[2] == 224
    img = read_img_ndarray["torchvision"](dog)
    assert img.size == 150528
    assert img.shape == (224, 224, 3)
