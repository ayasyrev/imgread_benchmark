from pathlib import Path
from typing import List, Optional, Union

IMG_EXT = (".jpeg", ".jpg", ".jfif", ".pjpeg", ".pjp", ".png", ".bmp", ".tif", ".tiff")


def get_img_filenames(
    data_dir: Union[str, Path],
    num_samples: Optional[int] = None,
    ext: Optional[str] = None,
) -> List[str]:
    """Return list of num_samples image filenames from data_dir.
    If no num_samples, return list of ALL images.

    Args:
        data_dir (str):
        num_samples (int, optional): Number of samples to return. Defaults to None.
        If num_samples not given return list of ALL images.


    Returns:
        List[str]: List of filenames
    """
    img_filenames = [
        str(fn) for fn in Path(data_dir).rglob("*.*") if fn.suffix.lower() in IMG_EXT
    ]
    if ext is not None:
        img_filenames = [item for item in img_filenames if ext in item]
    num_samples = num_samples or len(img_filenames)
    return img_filenames[:num_samples]
