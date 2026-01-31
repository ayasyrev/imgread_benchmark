from importlib.util import find_spec

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
        from jpeg4py import _cffi as jpeg4py_cffi

        lib = getattr(jpeg4py_cffi, "lib", None)
        if lib is None:
            initializer = getattr(jpeg4py_cffi, "_initialize", None)
            if initializer is not None:
                backends = getattr(jpeg4py_cffi, "backends", None)
                if backends is None:
                    initializer()
                else:
                    initializer(backends)
    except Exception:
        return False
    return True


def _build_img_lib_available() -> list[str]:
    available: list[str] = []
    for lib_name in lib_to_package:
        if find_spec(lib_name) is None:
            continue
        if lib_name == "jpeg4py" and not _is_jpeg4py_usable():
            continue
        available.append(lib_name)
    return available


img_lib_available = _build_img_lib_available()
