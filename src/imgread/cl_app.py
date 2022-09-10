from .img_libs import img_lib_available
from .version import __version__


def cl_app() -> None:
    print(f"imgread version: {__version__}")
    print("Availible image libs:")
    for img_lib in img_lib_available:
        print(" " * 4, img_lib)
