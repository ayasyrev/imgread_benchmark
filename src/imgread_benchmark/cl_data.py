from dataclasses import dataclass
from argparsecfg import field_argument
from argparsecfg.app import app
from .datasets import ImagenetteProvider

@dataclass
class DownloadConfig:
    dataset: str = field_argument(
        "dataset",
        choices=["imagenette"],
        help="Dataset to download"
    )
    size: str = field_argument(
        "--size",
        default="full",
        choices=["full", "320", "160"],
        help="Size of the dataset to download"
    )

@app(description="Download datasets for benchmarking")
def download_app(cfg: DownloadConfig):
    """Download a dataset."""
    if cfg.dataset == "imagenette":
        provider = ImagenetteProvider()
        provider.download(size=cfg.size)

if __name__ == "__main__":
    download_app()
