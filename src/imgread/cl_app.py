from argparsecfg import field_argument
from argparsecfg.app import app
from dataclasses import dataclass

from .read_img import img_libs
from .version import __version__


@dataclass
class AppConfig:
    version: bool = field_argument(
        "-v", "--version", help="App version.", default=False
    )


@app(
    description="Imgread. Helpers utils for check and benchmark libs for read image files.",
)
def imgread(
    cfg: AppConfig,
) -> None:
    """Imgread. Helpers utils for check and benchmark libs for read image files."""
    print("Imgread. Helpers utils for check and benchmark libs for read image files.")
    print(f"Available {len(img_libs)}s image libs:")

    if cfg.version:
        print(f"version: {__version__}")
    else:
        max_len = max(len(lib_name) for lib_name in img_libs)
        for img_lib in img_libs:
            print(f"    {img_lib:{max_len}} {img_libs[img_lib].__version__}")


if __name__ == "__main__":  # pragma: no cover
    imgread()
