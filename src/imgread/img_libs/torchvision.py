from torchvision.io import read_image
from torchvision import __version__  # noqa: F401


def read_img(img_path: str):
    img = read_image(img_path)
    return img
