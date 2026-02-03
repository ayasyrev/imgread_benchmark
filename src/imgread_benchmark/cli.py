from dataclasses import dataclass
from typing import Sequence, Union

from argparsecfg import field_argument
from argparsecfg.app import App

_KNOWN_COMMANDS = {"benchmark", "libs", "data"}


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    argv = list(argv)
    if not argv:
        return []
    first = argv[0]
    if first in _KNOWN_COMMANDS:
        return argv
    if first.startswith("-"):
        return argv
    return ["benchmark", *argv]


@dataclass
class RootConfig:
    version: bool = field_argument(
        "-V",
        "--version",
        default=False,
        action="store_true",
        help="Show version and exit.",
    )


def _root(cfg: RootConfig, cli: App) -> None:
    if cfg.version:
        from .version import __version__

        print(__version__)
        return
    cli.parser.print_help()


def _build_cli() -> App:
    cli = App(description="Benchmark image-reading libraries.")

    def _main_wrapper(cfg: RootConfig) -> None:
        _root(cfg, cli)

    cli.main(_main_wrapper)

    @dataclass
    class DataConfig:
        dataset: str = field_argument("dataset", help="Dataset to download")
        size: str = field_argument(
            "--size",
            default="full",
            choices=["full", "320", "160"],
            help="Size of the dataset to download",
        )

    def data(cfg: DataConfig) -> None:
        from .cl_data import DATASET_PROVIDERS

        provider_cls = DATASET_PROVIDERS.get(cfg.dataset)
        if provider_cls is None:
            raise SystemExit(2)
        provider_cls().download(size=cfg.size)

    cli.command(data)

    @dataclass
    class LibsConfig:
        pass

    def libs(cfg: LibsConfig) -> None:
        from .img_libs.img_libs_pkgs import get_img_lib_available
        from .read_img import get_read_img_version

        names = get_img_lib_available()
        versions = get_read_img_version()
        print(f"Available {len(names)} image libs:")
        if not names:
            return
        width = max(len(n) for n in names)
        for name in names:
            print(f"    {name:{width}} {versions.get(name, 'unknown')}")

    cli.command(libs)

    @dataclass
    class BenchmarkConfig:
        img_path: str = field_argument(
            "img_path", help="Directory with images for benchmark"
        )
        num_samples: int = field_argument(
            "-n", default=200, help="Number of samples for test, default 200."
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
            "-x", default=None, help="Image lib exclude from test"
        )
        multiprocessing: bool = field_argument(
            "-m",
            default=False,
            action="store_true",
            help="use multiprocessing, default=False",
        )
        nw: int = field_argument(default=None, help="num workers, if 0 -> use all cpus")

    def benchmark(cfg: BenchmarkConfig) -> None:
        from pathlib import Path as StdLibPath
        import sys as _sys

        from .get_img_filenames import get_img_filenames

        if not StdLibPath(cfg.img_path).exists():
            print(f"Img dir {cfg.img_path} dos not exist!")
            raise _sys.exit()
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

        from .benchmark import BenchmarkImgRead

        bench = BenchmarkImgRead(filenames=filenames, target_format=cfg.to)
        bench.run(
            func_name=cfg.img_lib,
            exclude=cfg.exclude,
            multiprocessing=cfg.multiprocessing,
            num_workers=cfg.nw,
        )

    cli.command(benchmark)
    return cli


_CLI: App | None = None


def _get_cli() -> App:
    global _CLI
    if _CLI is None:
        _CLI = _build_cli()
    return _CLI


def main(argv: Union[Sequence[str], None] = None) -> None:
    import sys as _sys

    if argv is None:
        argv = _sys.argv[1:]
    _get_cli()(_normalize_argv(argv))
