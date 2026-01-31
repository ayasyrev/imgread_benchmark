import pytest
from unittest.mock import MagicMock, patch

# This import is expected to fail initially
try:
    from imgread_benchmark.datasets import ImagenetteProvider
except ImportError:
    ImagenetteProvider = None


def test_imagenette_properties():
    """Test Imagenette specific properties."""
    if ImagenetteProvider is None:
        pytest.fail("ImagenetteProvider not implemented")
    provider = ImagenetteProvider()
    assert provider.name == "imagenette"
    assert provider.default_size == "full"


def test_imagenette_urls():
    """Test URL generation for different sizes."""
    if ImagenetteProvider is None:
        pytest.fail("ImagenetteProvider not implemented")
    provider = ImagenetteProvider()

    assert (
        provider.get_url("full")
        == "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    )
    assert (
        provider.get_url("320")
        == "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    )
    assert (
        provider.get_url("160")
        == "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    )


def test_imagenette_invalid_size():
    """Test that invalid sizes raise an error."""
    if ImagenetteProvider is None:
        pytest.fail("ImagenetteProvider not implemented")
    provider = ImagenetteProvider()
    with pytest.raises(ValueError):
        provider.get_url("invalid_size")


def test_imagenette_download_integration(tmp_path):
    """Test download logic with mocks."""
    if ImagenetteProvider is None:
        pytest.fail("ImagenetteProvider not implemented")
    provider = ImagenetteProvider(root_dir=tmp_path)

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        with patch("builtins.open", new_callable=MagicMock):
            with patch("tarfile.open"):
                provider.download("160")

    assert (provider.dataset_dir / ".ready").exists()
