from __future__ import annotations

from typing import Sequence

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
