import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from imgread_benchmark.cl_data import download_app

def test_integration_data_imagenette_160(tmp_path):
    """Integration test for downloading imagenette 160 (mocked network)."""
    def mock_init(self, root_dir=tmp_path):
        self.root_dir = Path(root_dir)
        self.dataset_dir = self.root_dir / "imagenette"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        return None

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_response
        
        with patch("builtins.open", new_callable=MagicMock):
            with patch("tarfile.open") as mock_tar:
                # We need to ensure we don't hit the real file system for the extraction logic
                # which calls archive_path.unlink()
                with patch("imgread_benchmark.datasets.Path.unlink"):
                    with patch("imgread_benchmark.cl_data.ImagenetteProvider.__init__", mock_init):
                         download_app(["imagenette", "--size", "160"])
    
    assert (tmp_path / "imagenette" / ".ready").exists()
