from __future__ import annotations
from typing import Callable, Dict, List, Literal, Optional

from benchmark_utils import BenchmarkIter

from .get_img_filenames import get_img_filenames
from .read_img import read_img, read_img_ndarray, read_img_pil

READ_TO_FORMAT = {
    "def": read_img,
    "pil": read_img_pil,
    "np": read_img_ndarray,
}

__all__ = ["BenchmarkImgRead"]


class BenchmarkImgRead(BenchmarkIter):
    """Benchmark image read."""

    def __init__(
        self,
        img_path: Optional[str] = None,
        num_samples: int = 0,
        target_format: Literal["def", "pil", "np"] = "def",
        func_dict: Optional[Dict[str, Callable]] = None,
        filenames: Optional[List[str]] = None,
        num_repeats: int = 5,
        clear_progress: bool = False,
    ):
        self._target_format = target_format
        func_to_test = func_dict or READ_TO_FORMAT[target_format]
        img_path = img_path or "."
        if filenames is None:
            filenames = get_img_filenames(img_path, num_samples=num_samples)
        super().__init__(
            func=func_to_test,
            item_list=filenames,
            num_repeats=num_repeats,
            clear_progress=clear_progress,
        )

    @property
    def target_format(self) -> Literal["def", "pil", "np"]:
        return self._target_format

    @target_format.setter
    def target_format(self, target_format: Literal["def", "pil", "np"]) -> None:
        if target_format not in READ_TO_FORMAT:
            print(f"{target_format} not in available format.")
            print("available: ", ", ".join(READ_TO_FORMAT.keys()))
        else:
            self._target_format = target_format
            self.func_dict = READ_TO_FORMAT[target_format]

    @property
    def func_names(self) -> str:
        return f"target_format: {self.target_format}, image libs: {', '.join(self.func_dict.keys())}"
