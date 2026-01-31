import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from imgread_benchmark.datasets import DatasetProvider


# Define a concrete implementation for testing the abstract base class
class MockDataset(DatasetProvider):
    @property
    def name(self) -> str:
        return "mock_dataset"

    @property
    def default_size(self) -> str:
        return "full"

    def get_url(self, size: str) -> str:
        return f"http://example.com/{size}.tgz"


class LateNameDataset(DatasetProvider):
    def __init__(self, root_dir: Path | str = ".data"):
        super().__init__(root_dir)
        self._name = "late_dataset"

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_size(self) -> str:
        return "full"

    def get_url(self, size: str) -> str:
        return f"http://example.com/{size}.tgz"


def test_dataset_provider_initialization(tmp_path):
    """Test that the base class initializes correctly with a root path."""
    provider = MockDataset(root_dir=tmp_path)
    assert provider.root_dir == tmp_path
    assert provider.dataset_dir.exists()


def test_dataset_dir_allows_late_name_init(tmp_path):
    provider = LateNameDataset(root_dir=tmp_path)

    dataset_dir = provider.dataset_dir

    assert dataset_dir == tmp_path / "late_dataset"
    assert dataset_dir.exists()


def test_dataset_provider_is_abstract():
    """Ensure the base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DatasetProvider(Path("."))


def test_download_skips_if_exists(tmp_path):
    """Test that download is skipped if the sentinel file exists."""
    provider = MockDataset(root_dir=tmp_path)
    # Create a fake sentinel file
    (provider.dataset_dir / ".ready").touch()

    with patch("requests.get") as mock_get:
        provider.download("full")
        mock_get.assert_not_called()


def test_download_executes_if_missing(tmp_path):
    """Test that download happens if dataset is missing."""
    provider = MockDataset(root_dir=tmp_path)

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        # We need to mock open since we don't want to actually write large files
        with patch("builtins.open", new_callable=MagicMock):
            # Mock extraction to avoid dealing with real tarfiles in this unit test
            with patch.object(provider, "extract") as mock_extract:
                provider.download("full")

    mock_get.assert_called_once()
    mock_extract.assert_called_once()
    assert (provider.dataset_dir / ".ready").exists()
