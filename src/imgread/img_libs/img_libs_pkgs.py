from importlib.util import find_spec


lib_to_package = {
    "PIL": "pillow",
    "accimage": "accimage",  # only conda
    "jpeg4py": "jpeg4py",
    "cv2": "opencv-python-headless",  # conda - opencv
    "skimage": "scikit-image",  # conda
    "imageio": "imageio",  # conda
    "pyvips": "pyvips",  # conda
    "torchvision": "torchvision",
}


img_lib_available = [
    lib_name for lib_name in lib_to_package if find_spec(lib_name) is not None
]
