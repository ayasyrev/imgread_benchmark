import numpy as np
import pytest
from unittest.mock import patch


def test_imagecodecs_adapter():
    pytest.importorskip("imagecodecs")
    from imgread_benchmark.img_libs import imagecodecs

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
