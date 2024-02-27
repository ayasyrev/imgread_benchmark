import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image
from torchvision import __version__  # noqa: F401
from torchvision import io

__all__ = [
    "__version__",
    "read_img",
    "read_img_ndarray",
    "read_img_pil",
]


def read_img(img_path: str) -> torch.Tensor:
    """Read image from path with torchvision.io. Returns torch.Tensor.

    Returns:
        torch.Tensor: Image as torch.Tensor
    """
    return io.read_image(img_path)


def read_img_ndarray(img_path: str) -> np.ndarray:
    """Reads image from path with torchvision.io and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    image = read_img(img_path)
    return np.transpose(image.numpy(), (1, 2, 0))


def read_img_pil(img_path: str) -> Image.Image:
    """Read image with torchvision.io and returns PIL.Image.

    Returns:
        PIL.Image: Image as PIL.Image
    """
    return F.to_pil_image(read_img(img_path))
