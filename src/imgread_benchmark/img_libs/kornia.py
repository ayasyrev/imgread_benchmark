from pathlib import Path
from typing import Literal

import kornia_rs
import numpy as np
import torch
from kornia import io
from PIL import Image

try:
    from kornia.image.utils import image_to_tensor, tensor_to_image
except ImportError:  # Kornia < 0.8.3
    from kornia.utils import image_to_tensor, tensor_to_image

__all__ = [
    "read_img",
    "read_img_ndarray",
    "read_img_pil",
]


def read_img(
    img_path: str,
    load_type: io.ImageLoadType = io.ImageLoadType.RGB8,
    device: Literal["cpu", "cuda"] = "cpu",
) -> torch.Tensor:
    """Read image from path with kornia.io. Returns torch.Tensor.

    Returns:
        torch.Tensor: Image as torch.Tensor
    """
    if (
        load_type == io.ImageLoadType.RGB8
        and Path(img_path).suffix.lower() in {".jpg", ".jpeg"}
        and not hasattr(kornia_rs, "read_image_jpegturbo")
    ):
        # kornia_rs 0.1.14 moved JPEG I/O; Kornia 0.8.3 still calls the old name.
        array = kornia_rs.io.read_image_jpegturbo(str(img_path))
        return image_to_tensor(array, keepdim=True).to(device=device)
    return io.load_image(img_path, desired_type=load_type, device=device)


def read_img_ndarray(img_path: str) -> np.ndarray:
    """Reads image from path with kornia.io and returns numpy array with shape (width, height, channels).

    Returns:
        np.ndarray: Image as numpy array with shape (width, height, channels)
    """
    return tensor_to_image(read_img(img_path))


def read_img_pil(img_path: str) -> Image.Image:
    """Read image with kornia.io and returns PIL.Image.

    Returns:
        PIL.Image: Image as PIL.Image
    """
    return Image.fromarray(read_img_ndarray(img_path))
