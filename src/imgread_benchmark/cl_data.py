from dataclasses import dataclass
from argparsecfg import field_argument
from argparsecfg.app import app
from .datasets import DatasetProvider, ImagenetteProvider

DATASET_PROVIDERS: dict[str, type[DatasetProvider]] = {
    "imagenette": ImagenetteProvider,
}


@dataclass
class DownloadConfig:
    dataset: str = field_argument(
        "dataset", choices=list(DATASET_PROVIDERS), help="Dataset to download"
    )
    size: str = field_argument(
        "--size",
        default="full",
        choices=["full", "320", "160"],
        help="Size of the dataset to download",
    )


@app(description="Download datasets for benchmarking")
def download_app(cfg: DownloadConfig):
    """Download a dataset."""
    provider_cls = DATASET_PROVIDERS.get(cfg.dataset)
    if provider_cls is None:
        return
    provider = provider_cls()
    provider.download(size=cfg.size)


if __name__ == "__main__":
    download_app()
