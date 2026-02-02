from importlib.util import find_spec
from pathlib import Path
import tempfile
from functools import lru_cache

lib_to_package = {
    "PIL": "pillow",
    "accimage": "accimage",  # only conda
    "jpeg4py": "jpeg4py",
    "cv2": "opencv-python-headless",  # conda - opencv
    "skimage": "scikit-image",  # conda
    "imageio": "imageio",  # conda
    "imread": "imread",  # conda
    "kornia": "kornia",
    # "pyvips": "pyvips",  # conda
    "torchvision": "torchvision",
}


def _is_jpeg4py_usable() -> bool:
    try:
        import jpeg4py

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
            result = jpeg4py.JPEG(tmp_path).decode()
            del result
            return True
        except (AttributeError, OSError, ValueError) as e:
            error_str = str(e)
            if "decompressor" in error_str:
                return False
            import warnings

            warnings.warn(f"jpeg4py test failed: {e}", RuntimeWarning)
            return False
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        import warnings

        warnings.warn(f"jpeg4py not available: {e}", RuntimeWarning)
        return False


@lru_cache(maxsize=1)
def get_img_lib_available() -> list[str]:
    """Get list of available image libraries (lazy, cached)."""
    available: list[str] = []
    for lib_name in lib_to_package:
        if find_spec(lib_name) is None:
            continue
        if lib_name == "jpeg4py" and not _is_jpeg4py_usable():
            continue
        available.append(lib_name)
    return available


def _build_img_lib_available() -> list[str]:
    available: list[str] = []
    for lib_name in lib_to_package:
        if find_spec(lib_name) is None:
            continue
        if lib_name == "jpeg4py" and not _is_jpeg4py_usable():
            continue
        available.append(lib_name)
    return available


# For backwards compatibility - deprecated, use get_img_lib_available()
img_lib_available = get_img_lib_available()
