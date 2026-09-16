import abc
import tarfile
from pathlib import Path
import requests
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
)

console = Console()


class DatasetProvider(abc.ABC):
    """Abstract base class for dataset providers."""

    def __init__(self, root_dir: Path | str = ".data"):
        self.root_dir = Path(root_dir)

    @property
    def dataset_dir(self) -> Path:
        path = self.root_dir / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

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
        url = self.get_url(size)
        archive_name = Path(url).name
        expected_dir = self.dataset_dir / Path(archive_name).stem

        sentinel = self.dataset_dir / ".ready"
        if sentinel.exists() and expected_dir.exists():
            contents = [
                p.name for p in self.dataset_dir.iterdir() if p.name != ".ready"
            ]
            console.print(
                f"[green]✓[/green] {self.name} dataset already downloaded ({size})."
            )
            if contents:
                console.print(f"  Available: {', '.join(contents)}")
            return

        archive_path = self.dataset_dir / archive_name

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Downloading {self.name} ({size})...", total=total_size
            )
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

        try:
            self.extract(archive_path)
        finally:
            archive_path.unlink(missing_ok=True)  # Remove archive after extraction
        sentinel.touch()
        console.print(
            f"[green]✓[/green] {self.name} dataset downloaded successfully ({size})."
        )

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
                members = tar.getmembers()
                dataset_root = self.dataset_dir.resolve()
                for member in members:
                    member_path = (self.dataset_dir / member.name).resolve()
                    if not member_path.is_relative_to(dataset_root):
                        raise IOError(
                            f"Attempted path traversal in tar file: {member.name}"
                        )
                tar.extractall(path=self.dataset_dir, members=members)


class ImagenetteProvider(DatasetProvider):
    """Dataset provider for Imagenette."""

    @property
    def name(self) -> str:
        return "imagenette"

    @property
    def default_size(self) -> str:
        return "full"

    def get_url(self, size: str) -> str:
        base_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2"
        if size == "full":
            return f"{base_url}.tgz"
        elif size in ["320", "160"]:
            return f"{base_url}-{size}.tgz"
        else:
            raise ValueError(
                f"Invalid size: {size}. valid sizes are 'full', '320', '160'."
            )
