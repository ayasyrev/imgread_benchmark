from dataclasses import MISSING, field as dataclass_field
import sys
from typing import Any

from argparsecfg.core import add_argument_metadata


def field_argument(
    *name_or_flags: str,
    default: Any = MISSING,
    default_factory: Any = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: dict[str, Any] | None = None,
    kw_only: Any = MISSING,
    flag: str | None = None,
    action: str | None = None,
    nargs: int | str | None = None,
    const: Any = None,
    type: str | type | None = None,
    choices: list[Any] | tuple[Any, ...] | None = None,
    required: bool | None = None,
    help: str | None = None,
    metavar: str | None = None,
    dest: str | None = None,
    version: str | None = None,
) -> Any:
    """Drop-in replacement for argparsecfg.field_argument on Python 3.14+."""
    del version
    arg_metadata = add_argument_metadata(
        *name_or_flags,
        flag=flag,
        action=action,
        nargs=nargs,
        const=const,
        type=type,
        choices=choices,
        required=required,
        help=help,
        metavar=metavar,
        dest=dest,
    )
    field_kwargs = {
        "default": default,
        "default_factory": default_factory,
        "init": init,
        "repr": repr,
        "hash": hash,
        "compare": compare,
    }

    if metadata is not None:
        arg_metadata.update(metadata)
    field_kwargs["metadata"] = arg_metadata

    return dataclass_field(**field_kwargs)
