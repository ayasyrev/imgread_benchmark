import pytest
from pathlib import Path

from imgread.read_img import read_img_pil, read_img_nparray, read_img
from imgread.img_libs.img_libs_pkgs import lib_to_package, img_lib_available


if "torchvision" in img_lib_available:
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
def test_read_img_nparray(img_lib):
    assert img_lib in lib_to_package
    img = read_img_nparray[img_lib](dog)
    assert img.size == 150528


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
    img = read_img_nparray["torchvision"](dog)
    assert img.size == 150528
