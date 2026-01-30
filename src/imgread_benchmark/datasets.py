import abc
import tarfile
from pathlib import Path
import requests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

class DatasetProvider(abc.ABC):
    """Abstract base class for dataset providers."""

    def __init__(self, root_dir: Path | str = ".data"):
        self.root_dir = Path(root_dir)
        self.dataset_dir = self.root_dir / self.name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the dataset."""

    @property
    @abc.abstractmethod
    def default_size(self) -> str:
        """Default size/version of the dataset."""

    @abc.abstractmethod
    def get_url(self, size: str) -> str:
        """Get the download URL for a specific size."""

    def download(self, size: str | None = None):
        """Download and extract the dataset."""
        size = size or self.default_size
        sentinel = self.dataset_dir / ".ready"
        if sentinel.exists():
            return

        url = self.get_url(size)
        archive_path = self.dataset_dir / Path(url).name

        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task(f"Downloading {self.name} ({size})...", total=total_size)
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

        self.extract(archive_path)
        archive_path.unlink(missing_ok=True)  # Remove archive after extraction
        sentinel.touch()

    def extract(self, archive_path: Path):
        """Extract the dataset archive."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            progress.add_task(f"Extracting {archive_path.name}...", total=None)
            # Check if archive exists before trying to open it
            if not archive_path.exists():
                return
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self.dataset_dir)
