from typing import Callable, Dict, List, Optional

from benchmark_utils import BenchmarkIter

from .get_img_filenames import get_img_filenames
from .read_img import read_img, read_img_nparray, read_img_pil

read_to_format = {
    "def": read_img,
    "pil": read_img_pil,
    "np": read_img_nparray,
}


class BenchmarkImgRead(BenchmarkIter):
    """Benchmark image read."""

    def __init__(
        self,
        img_path: Optional[str] = None,
        num_samples: int = 0,
        to_format: str = "def",
        func_dict: Optional[Dict[str, Callable]] = None,
        filenames: Optional[List[str]] = None,
        num_repeats: int = 5,
        clear_progress: bool = False,
    ):
        self.to_format = to_format
        func_to_test = func_dict or read_to_format[to_format]
        img_path = img_path or "."
        if filenames is None:
            filenames = get_img_filenames(img_path, num_samples=num_samples)
        super().__init__(
            func=func_to_test,
            item_list=filenames,
            num_repeats=num_repeats,
            clear_progress=clear_progress,
        )

    def set_func_dict(self, to_format: str) -> None:
        if to_format not in read_to_format:
            print(f"{to_format} not in available format.")
            print("available: ", ", ".join(read_to_format.keys()))
        else:
            self.to_format = to_format
            self.func_dict = read_to_format[to_format]

    @property
    def func_names(self) -> str:
        return f"to {self.to_format}: {', '.join(self.func_dict.keys())}"
