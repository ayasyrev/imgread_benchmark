from typing import Literal

import kornia as K
import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from kornia import __version__  # noqa: F401
from kornia import io
from PIL import Image

__all__ = [
    "__version__",
    "read_img",
    "read_img_ndarray",
    "read_img_pil",
]


def read_img(
    img_path: str,
    load_type: io.ImageLoadType = io.ImageLoadType.UNCHANGED,
    device: Literal["cpu", "cuda"] = "cpu",
) -> torch.Tensor:
    """Read image from path with kornia.io. Returns torch.Tensor.

    Returns:
        torch.Tensor: Image as torch.Tensor
    """
    return io.load_image(img_path, desired_type=load_type, device=device)


def read_img_ndarray(img_path: str) -> np.ndarray:
    """Reads image from path with kornia.io and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    return K.utils.tensor_to_image(read_img(img_path))


def read_img_pil(img_path: str) -> Image.Image:
    """Read image with kornia.io and returns PIL.Image.

    Returns:
        PIL.Image: Image as PIL.Image
    """
    return F.to_pil_image(read_img(img_path))
