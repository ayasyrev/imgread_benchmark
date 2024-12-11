from argparsecfg import field_argument
from argparsecfg.app import app
from dataclasses import dataclass

from .read_img import img_libs, read_img_version
from .version import __version__


@dataclass
class AppConfig:
    version: bool = field_argument(
        "-v", "--version", help="App version.", default=False
    )


@app(
    description="Imgread_benchmark. Helpers utils for check and benchmark libs for read image files.",
)
def imgread_versions(
    cfg: AppConfig,
) -> None:
    """Imgread_benchmark. Helpers utils for check and benchmark libs for read image files."""
    print("Imgread. Helpers utils for check and benchmark libs for read image files.")
    print(f"Available {len(img_libs)} image libs:")

    if cfg.version:
        print(f"version: {__version__}")
    else:
        max_len = max(len(lib_name) for lib_name in img_libs)
        for img_lib in img_libs:
            print(f"    {img_lib:{max_len}} {read_img_version[img_lib]}")


if __name__ == "__main__":  # pragma: no cover
    imgread_versions()
