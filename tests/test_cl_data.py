import pytest
from unittest.mock import patch
from imgread_benchmark.cl_data import download_app


def test_data_download_args():
    """Test that the data app correctly parses download arguments."""
    with patch("imgread_benchmark.cl_data.ImagenetteProvider") as mock_provider_cls:
        mock_instance = mock_provider_cls.return_value
        with patch.dict(
            "imgread_benchmark.cl_data.DATASET_PROVIDERS",
            {"imagenette": mock_provider_cls},
        ):
            # Test downloading imagenette with specific size
            # ['cl_data.py', 'imagenette', '--size', '160']
            download_app(["imagenette", "--size", "160"])

            mock_instance.download.assert_called_once_with(size="160")


def test_data_download_default_size():
    """Test that the data app uses default size if none provided."""
    with patch("imgread_benchmark.cl_data.ImagenetteProvider") as mock_provider_cls:
        mock_instance = mock_provider_cls.return_value
        with patch.dict(
            "imgread_benchmark.cl_data.DATASET_PROVIDERS",
            {"imagenette": mock_provider_cls},
        ):
            download_app(["imagenette"])

            mock_instance.download.assert_called_once_with(size="full")


def test_data_invalid_dataset():
    """Test that invalid dataset name raises SystemExit (from argparse)."""
    with pytest.raises(SystemExit):
        download_app(["invalid_ds"])


def test_dataset_provider_map_contains_imagenette():
    from imgread_benchmark import cl_data

    assert "imagenette" in cl_data.DATASET_PROVIDERS
    assert cl_data.DATASET_PROVIDERS["imagenette"].__name__ == "ImagenetteProvider"
