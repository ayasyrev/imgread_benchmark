import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

from imgread_benchmark.img_libs import ajpegli, imagecodecs, simplejpeg, turbojpeg

def test_ajpegli_adapter():
    with patch("ajpegli.imread") as mock_imread:
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        
        res = ajpegli.read_img("fake.jpg")
        mock_imread.assert_called_once_with("fake.jpg", mode="RGB")
        assert res.shape == (10, 10, 3)
        
        res_pil = ajpegli.read_img_pil("fake.jpg")
        assert isinstance(res_pil, Image.Image)
        assert res_pil.size == (10, 10)

def test_simplejpeg_adapter():
    with patch("builtins.open", MagicMock()):
        with patch("simplejpeg.decode_jpeg") as mock_decode:
            mock_decode.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
            
            res = simplejpeg.read_img("fake.jpg")
            assert res.shape == (10, 10, 3)
            
            res_pil = simplejpeg.read_img_pil("fake.jpg")
            assert isinstance(res_pil, Image.Image)

def test_turbojpeg_adapter():
    with patch("builtins.open", MagicMock()):
        with patch("turbojpeg.decompress") as mock_decompress:
            # We need something that np.array() can handle.
            # A real numpy array will do for the mock return.
            fake_arr = np.zeros((10, 10, 3), dtype=np.uint8)
            mock_decompress.return_value = fake_arr
            
            # Note: in reality turbojpeg.decompress returns TjImage,
            # but for the mock, if we return something np.array() accepts, it's fine.
            # Actually, if we return a numpy array, np.array(fake_arr) is still fake_arr.
            
            res = turbojpeg.read_img("fake.jpg")
            mock_decompress.assert_called_once()
            assert res.shape == (10, 10, 3)

def test_imagecodecs_adapter():
    with patch("imagecodecs.imread") as mock_imread:
        # Test RGB
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        assert imagecodecs.read_img("fake.jpg").shape == (10, 10, 3)
        
        # Test Grayscale
        mock_imread.return_value = np.zeros((10, 10), dtype=np.uint8)
        res = imagecodecs.read_img("fake.jpg")
        assert res.shape == (10, 10, 3)
        
        # Test RGBA
        mock_imread.return_value = np.zeros((10, 10, 4), dtype=np.uint8)
        res = imagecodecs.read_img("fake.jpg")
        assert res.shape == (10, 10, 3)
