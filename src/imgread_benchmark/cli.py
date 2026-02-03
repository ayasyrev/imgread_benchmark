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
