import numpy as np
import pytest
from PIL import Image
from imgread_benchmark.img_libs.img_libs_pkgs import img_lib_available, load_img_lib_adapter

dog = "tests/test_imgs/dog.jpg"

@pytest.mark.parametrize("lib_name", img_lib_available)
def test_standard_interface_pil(lib_name):
    try:
        lib = load_img_lib_adapter(lib_name)
    except Exception as e:
        pytest.skip(f"Could not load {lib_name}: {e}")
    
    if not hasattr(lib, "read_img_pil"):
        pytest.skip(f"{lib_name} does not have read_img_pil")
        
    img = lib.read_img_pil(dog)
    assert isinstance(img, Image.Image)
    assert img.size == (224, 224)

@pytest.mark.parametrize("lib_name", img_lib_available)
def test_standard_interface_ndarray(lib_name):
    try:
        lib = load_img_lib_adapter(lib_name)
    except Exception as e:
        pytest.skip(f"Could not load {lib_name}: {e}")
        
    if not hasattr(lib, "read_img_ndarray"):
        pytest.skip(f"{lib_name} does not have read_img_ndarray")
        
    img = lib.read_img_ndarray(dog)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3
    assert img.shape[2] == 3  # RGB
    assert img.shape[0] == 224
    assert img.shape[1] == 224
