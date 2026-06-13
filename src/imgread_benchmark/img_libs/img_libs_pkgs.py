import collections.abc
import sys
from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points
from importlib.util import find_spec
from pathlib import Path
import tempfile
from ctypes.util import find_library
from typing import Any

ENTRY_POINT_GROUP = "imgread_benchmark.img_libs"

_CORE_BUILTIN_LIB_TO_PACKAGE = {
    "PIL": "pillow",
    "accimage": "accimage",  # only conda
    "ajpegli": "ajpegli",
    "imagecodecs": "imagecodecs",
    "jpeg4py": "jpeg4py",
    "simplejpeg": "simplejpeg",
    "turbojpeg": "turbojpeg",
    "cv2": "opencv-python-headless",  # conda - opencv
    "skimage": "scikit-image",  # conda
    "imageio": "imageio",  # conda
    "imread": "imread",  # conda
    "kornia": "kornia",
    # "pyvips": "pyvips",  # conda
    "torchvision": "torchvision",
}

_ADDITIONAL_LIB_TO_PACKAGE = {
    # Optional non-core backends that should be appended after core built-ins.
    "local_rs": "local_rs",
    "imgread_rs": "imgread-rs",
}

_BUILTIN_LIB_TO_PACKAGE = {
    **_CORE_BUILTIN_LIB_TO_PACKAGE,
    **_ADDITIONAL_LIB_TO_PACKAGE,
}

# Backwards compatibility for code importing this constant directly.
lib_to_package = dict(_BUILTIN_LIB_TO_PACKAGE)


def _iter_builtin_libs_in_order() -> tuple[str, ...]:
    return (*_CORE_BUILTIN_LIB_TO_PACKAGE, *_ADDITIONAL_LIB_TO_PACKAGE)


def _has_jpeg_turbo() -> bool:
    return (
        find_library("turbojpeg") is not None
        or find_library("jpeg") is not None
        or find_library("libjpeg") is not None
    )


def _patch_jpeg4py_del(jpeg4py_module: object) -> None:
    jpeg_cls = getattr(jpeg4py_module, "JPEG", None)
    if jpeg_cls is None:
        return
    orig_del = getattr(jpeg_cls, "__del__", None)
    if orig_del is None or getattr(orig_del, "__imgread_safe__", False):
        return

    def safe_del(self) -> None:
        try:
            orig_del(self)
        except AttributeError:
            return None

    safe_del.__imgread_safe__ = True
    setattr(jpeg_cls, "__del__", safe_del)


def _is_jpeg4py_usable() -> bool:
    try:
        import jpeg4py

        _patch_jpeg4py_del(jpeg4py)

        if not _has_jpeg_turbo():
            return False

        minimal_jpeg = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00\x43\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01"
            b"\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01"
            b"\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01"
            b"\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0"
            b'\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff'
            b"\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xda\x00"
            b"\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfd\x9f\xff\xd9"
        )

        tmp_path = None
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(minimal_jpeg)
            tmp_path = tmp_file.name

        try:
            jpeg4py.JPEG(tmp_path).decode()
            return True
        except (AttributeError, OSError, ValueError):
            return False
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        return False


def _iter_entry_points() -> list[EntryPoint]:
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
        return list(eps)
    except TypeError:
        eps = entry_points()
        return list(eps.get(ENTRY_POINT_GROUP, ()))
    except Exception:
        return []


def _entry_point_module_exists(ep: EntryPoint) -> bool:
    module = getattr(ep, "module", None)
    if not module:
        return True
    try:
        return find_spec(module.split(".")[0]) is not None
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def get_plugin_entry_points() -> dict[str, EntryPoint]:
    """Get discovered external plugin entry points by backend name."""
    plugins: dict[str, EntryPoint] = {}
    for ep in sorted(_iter_entry_points(), key=lambda item: (item.name, item.value)):
        if ep.name in _BUILTIN_LIB_TO_PACKAGE:
            continue
        if ep.name in plugins:
            continue
        if not _entry_point_module_exists(ep):
            continue
        plugins[ep.name] = ep
    return plugins


def load_img_lib_adapter(lib_name: str) -> Any:
    """Load built-in adapter module or external entry-point adapter."""
    if lib_name in _BUILTIN_LIB_TO_PACKAGE:
        import importlib

        return importlib.import_module(f"imgread_benchmark.img_libs.{lib_name}")
    ep = get_plugin_entry_points().get(lib_name)
    if ep is None:
        raise ModuleNotFoundError(f"Unknown image backend: {lib_name}")
    return ep.load()


def _plugin_is_available(ep: EntryPoint) -> bool:
    try:
        adapter = ep.load()
    except Exception as exc:
        print(
            f"Warning: Could not load plugin entry point '{ep.name}': {exc}",
            file=sys.stderr,
        )
        return False
    is_available = getattr(adapter, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return True


@lru_cache(maxsize=1)
def get_lib_package_map() -> dict[str, str]:
    """Get backend name -> package name map for built-ins and plugins."""
    packages = dict(_BUILTIN_LIB_TO_PACKAGE)
    for name, ep in get_plugin_entry_points().items():
        dist = getattr(ep, "dist", None)
        dist_name = getattr(dist, "name", None)
        packages[name] = dist_name or ep.module.split(".")[0]
    return packages


@lru_cache(maxsize=1)
def get_img_lib_available() -> list[str]:
    """Get list of available image libraries (lazy, cached)."""
    available: list[str] = []
    for lib_name in _iter_builtin_libs_in_order():
        if find_spec(lib_name) is None:
            continue
        if lib_name == "jpeg4py" and not _is_jpeg4py_usable():
            continue
        try:
            load_img_lib_adapter(lib_name)
        except Exception:
            continue
        available.append(lib_name)
    for lib_name, ep in get_plugin_entry_points().items():
        if _plugin_is_available(ep):
            available.append(lib_name)
    return available


# For backwards compatibility - deprecated, use get_img_lib_available()
class _LazyList(collections.abc.Sequence):
    def __init__(self, factory):
        self._factory = factory
        self._value: list[str] | None = None

    def _get(self) -> list[str]:
        if self._value is None:
            self._value = list(self._factory())
        return self._value

    def __len__(self) -> int:
        return len(self._get())

    def __getitem__(self, index):
        return self._get()[index]

    def __repr__(self) -> str:
        return repr(self._get())

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


img_lib_available = _LazyList(get_img_lib_available)
