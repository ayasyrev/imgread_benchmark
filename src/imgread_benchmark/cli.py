from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence, Union

from argparsecfg.app import App
from .argparse_compat import field_argument

_KNOWN_COMMANDS = {"benchmark", "libs", "data", "dataloader"}

_BENCHMARK_FLAG_ALLOWLIST = {
    "-n",
    "--num_samples",
    "-t",
    "--to",
    "-A",
    "--all",
    "-l",
    "--img_lib",
    "-x",
    "--exclude",
    "-m",
    "--multiprocessing",
    "--nw",
    "-r",
    "--repeats",
    "--shuffle",
    "--no-warmup",
}

_ROOT_ONLY_FLAGS = {"-h", "--help", "-V", "--version"}

_FLAGS_WITH_VALUES = {
    "-n",
    "--num_samples",
    "-t",
    "--to",
    "-l",
    "--img_lib",
    "-x",
    "--exclude",
    "--nw",
    "-r",
    "--repeats",
}


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    argv = list(argv)
    if not argv:
        return []

    first = argv[0]

    # Known commands pass through unchanged
    if first in _KNOWN_COMMANDS:
        return argv

    # Root-only flags pass through unchanged
    if first in _ROOT_ONLY_FLAGS:
        return argv

    # If first arg starts with '-', scan for allowlisted benchmark flags
    if first.startswith("-"):
        i = 0
        saw_allowlist = False

        while i < len(argv) and argv[i].startswith("-"):
            arg = argv[i]

            # Root-only flag means stop processing
            if arg in _ROOT_ONLY_FLAGS:
                return argv

            # Unknown flag means stop processing
            if arg not in _BENCHMARK_FLAG_ALLOWLIST:
                return argv

            # Flag is in allowlist
            saw_allowlist = True

            # If flag takes a value, consume it
            if arg in _FLAGS_WITH_VALUES:
                i += 1
                if i >= len(argv):
                    # Missing value - return unchanged
                    return argv

            i += 1

        # If we saw allowlisted flags, inject "benchmark"
        if saw_allowlist:
            return ["benchmark", *argv]

        # No allowlisted flags found - return unchanged
        return argv

    # Positional argument - inject "benchmark"
    return ["benchmark", *argv]


def _select_funcs_for_run(
    func_dict: dict[str, Callable[[str], object]],
    func_name: str | None,
    exclude: str | None,
) -> dict[str, Callable[[str], object]]:
    if func_name:
        if func_name in func_dict:
            return {func_name: func_dict[func_name]}
        return {}
    if exclude:
        return {name: func for name, func in func_dict.items() if name != exclude}
    return dict(func_dict)


def _get_multiprocessing_compat_errors(
    func_dict: dict[str, Callable[[str], object]],
    func_name: str | None,
    exclude: str | None,
) -> list[tuple[str, Exception]]:
    from benchmark_utils.benchmark import try_run
    from multiprocessing.reduction import ForkingPickler

    errors: list[tuple[str, Exception]] = []
    for name, func in _select_funcs_for_run(func_dict, func_name, exclude).items():
        try:
            ForkingPickler.dumps(partial(try_run, func))
        except Exception as exc:  # pragma: no cover - exercised via CLI tests
            errors.append((name, exc))
    return errors


def _probe_multiprocessing_workers(num_workers: int | None) -> None:
    from multiprocessing import Pool, cpu_count

    cpu_num = cpu_count() or 1
    workers = cpu_num if num_workers is None else num_workers
    if workers <= 0:
        workers = cpu_num
    with Pool(workers) as pool:
        pool.map(int, [1])


def _print_multiprocessing_start_error(exc: Exception) -> None:
    import sys as _sys

    print("Error: failed to start multiprocessing workers.", file=_sys.stderr)
    print(f"Detail: {type(exc).__name__}: {exc}", file=_sys.stderr)
    print(
        "Try running without -m/--multiprocessing or adjust environment multiprocessing permissions.",
        file=_sys.stderr,
    )


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
        import sys

        provider_cls = DATASET_PROVIDERS.get(cfg.dataset)
        if provider_cls is None:
            print(
                f"Error: Unknown dataset '{cfg.dataset}'. Available: {', '.join(DATASET_PROVIDERS)}",
                file=sys.stderr,
            )
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
            "-A",
            default=False,
            action="store_true",
            help="Use all images from folder",
        )
        img_lib: str = field_argument(
            "-l",
            "--img_lib",
            default=None,
            help="Image lib to test",
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
        nw: int = field_argument(
            default=None,
            help="num workers, if 0 -> use all cpus",
        )
        repeats: int = field_argument(
            "-r",
            "--repeats",
            default=5,
            help="Number of repeat runs, default 5",
        )
        shuffle: bool = field_argument(
            "--shuffle",
            default=False,
            action="store_true",
            help="Shuffle files before every repeat of each image reader",
        )
        no_warmup: bool = field_argument(
            flag="--no-warmup",
            default=False,
            action="store_true",
            help="Skip reading all selected files before the first timed benchmark",
        )

    def benchmark(cfg: BenchmarkConfig) -> None:
        from pathlib import Path as StdLibPath
        import sys as _sys

        from .get_img_filenames import get_img_filenames

        if not StdLibPath(cfg.img_path).exists():
            print(f"Error: Img dir '{cfg.img_path}' does not exist!", file=_sys.stderr)
            raise SystemExit(1)
        if cfg.nw is not None and cfg.nw < 0:
            print("Error: --nw must be a non-negative integer.", file=_sys.stderr)
            raise SystemExit(2)
        num_workers = None if cfg.nw == 0 else cfg.nw
        if cfg.all:
            cfg.num_samples = 0
        filenames = get_img_filenames(cfg.img_path, num_samples=cfg.num_samples)
        if not filenames:
            print(f"Error: No images found in '{cfg.img_path}'!", file=_sys.stderr)
            raise SystemExit(1)

        from .benchmark import BenchmarkImgRead, FileWarmupError

        bench = BenchmarkImgRead(
            filenames=filenames,
            target_format=cfg.to,
            num_repeats=cfg.repeats,
            shuffle=cfg.shuffle,
            warmup=not cfg.no_warmup,
        )
        if cfg.multiprocessing:
            compat_errors = _get_multiprocessing_compat_errors(
                bench.func_dict,
                cfg.img_lib,
                cfg.exclude,
            )
            if compat_errors:
                print(
                    "Error: selected image readers are not compatible with multiprocessing serialization.",
                    file=_sys.stderr,
                )
                for name, exc in compat_errors:
                    print(
                        f"  - {name}: {type(exc).__name__}: {exc}",
                        file=_sys.stderr,
                    )
                print(
                    "Try running without -m/--multiprocessing or use pickle-safe backend callables.",
                    file=_sys.stderr,
                )
                raise SystemExit(2)
            try:
                _probe_multiprocessing_workers(num_workers)
            except (PermissionError, RuntimeError, OSError) as exc:
                _print_multiprocessing_start_error(exc)
                raise SystemExit(2) from exc

        if len(filenames) < cfg.num_samples:
            print(
                f"! Number of files in {cfg.img_path}: {len(filenames)} less than num_samples: {cfg.num_samples}"
            )

        print(f"Benchmarking with images from {cfg.img_path}, target format: {cfg.to}")
        if cfg.num_samples:
            print(f"number of samples: {cfg.num_samples}")
        else:
            print(f"{len(filenames)} images.")

        try:
            bench.run(
                func_name=cfg.img_lib,
                exclude=cfg.exclude,
                multiprocessing=cfg.multiprocessing,
                num_workers=num_workers,
            )
        except FileWarmupError as exc:
            print(f"Error: {exc}", file=_sys.stderr)
            raise SystemExit(1) from exc
        except (PermissionError, RuntimeError, OSError) as exc:
            if not cfg.multiprocessing:
                raise
            _print_multiprocessing_start_error(exc)
            raise SystemExit(2) from exc

    cli.command(benchmark)
    from .dataloader.cli import dataloader

    cli.command(dataloader)
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
