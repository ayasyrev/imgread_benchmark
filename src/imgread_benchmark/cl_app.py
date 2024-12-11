from pathlib import Path
import sys
from argparsecfg import field_argument
from argparsecfg.app import app
from dataclasses import dataclass
from .benchmark import BenchmarkImgRead
from .get_img_filenames import get_img_filenames


@dataclass
class AppConfig:
    img_path: str = field_argument(
        "img_path",  # positional argument
        help="Directory with images for benchmark",
    )
    num_samples: int = field_argument(
        "-n",
        default=200,
        help="Number of samples for test, default 200.",
    )
    to: str = field_argument(
        "-t",
        default="def",
        help="Format for read image to: default: 'def', Pil: 'pil', or Numpy: 'np'.",
    )
    all: bool = field_argument(
        "-A", default=False, action="store_true", help="Use all images from folder"
    )
    img_lib: str = field_argument(
        "-l", "--img_lib", default=None, help="Image lib to test"
    )
    exclude: str = field_argument(
        "-x",
        default=None,
        help="Image lib exclude from test",
    )
    multiprocessing: bool = field_argument(
        "-m",
        default=False,
        action="store_true",
        help="use multiprocessing, default=False",
    )
    nw: int = field_argument(default=None, help="num workers, if 0 -> use all cpus")


@app(
    description="Benchmark read image functions.",
)
def benchmark(
    cfg: AppConfig,
) -> None:
    """Benchmark read image functions."""
    if not Path(cfg.img_path).exists():
        print(f"Img dir {cfg.img_path} dos not exist!")
        raise sys.exit()
    if cfg.all:
        cfg.num_samples = 0
    filenames = get_img_filenames(cfg.img_path, num_samples=cfg.num_samples)
    if len(filenames) < cfg.num_samples:
        print(
            f"! Number of files in {cfg.img_path}: {len(filenames)} less than num_samples: {cfg.num_samples}"
        )

    print(f"Benchmarking with images from {cfg.img_path}, target format: {cfg.to}")
    if cfg.num_samples:
        print(f"number of samples: {cfg.num_samples}")
    else:
        print(f"{len(filenames)} images.")

    bench = BenchmarkImgRead(
        filenames=filenames,
        target_format=cfg.to,
    )
    bench.run(
        func_name=cfg.img_lib,
        exclude=cfg.exclude,
        multiprocessing=cfg.multiprocessing,
        num_workers=cfg.nw,
    )


if __name__ == "__main__":  # pragma: no cover
    benchmark()
