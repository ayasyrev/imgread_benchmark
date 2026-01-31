from unittest.mock import MagicMock, patch
from imgread_benchmark.datasets import DatasetProvider


class MockDataset(DatasetProvider):
    @property
    def name(self) -> str:
        return "mock_dataset"

    @property
    def default_size(self) -> str:
        return "full"

    def get_url(self, size: str) -> str:
        return "http://example.com/file.tgz"


def test_progress_bar_usage(tmp_path):
    """Test that rich Progress is initialized during download."""
    provider = MockDataset(root_dir=tmp_path)

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response

        with patch("builtins.open", new_callable=MagicMock):
            with patch("imgread_benchmark.datasets.Progress") as mock_progress:
                # Need to mock the context manager return value
                mock_progress.return_value.__enter__.return_value = MagicMock()

                # Also mock extraction to avoid second progress bar interference in this specific test
                with patch.object(provider, "extract"):
                    provider.download("full")

    # Verify Progress was instantiated
    mock_progress.assert_called()
