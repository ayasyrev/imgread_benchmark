from typing import Callable, Dict, List, Optional, Union

from benchmark_utils import BenchmarkIter

from .read_img import read_img, read_img_nparray, read_img_pil
from .get_img_filenames import get_img_filenames


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
        img_lib_name: Union[str, List[str], None] = None,
        exclude_name: Union[str, List[str], None] = None,
        func_dict: Optional[Dict[str, Callable]] = None,
        filenames: Optional[List[str]] = None,
        num_repeats: int = 5,
        clear_progress: bool = False,
    ):

        func_to_test = func_dict or read_to_format[to_format]

        if img_lib_name is not None:
            if isinstance(img_lib_name, str):
                img_lib_name = [img_lib_name]
            func_to_test = {
                lib_name: func_to_test[lib_name] for lib_name in img_lib_name
            }
        elif exclude_name is not None:
            if isinstance(exclude_name, str):
                exclude_name = [exclude_name]
            for lib_name in exclude_name:
                func_to_test.pop(lib_name, None)

        img_path = img_path or "."
        if filenames is None:
            filenames = get_img_filenames(img_path, num_samples=num_samples)
        super().__init__(
            func=func_to_test,
            item_list=filenames,
            num_repeats=num_repeats,
            clear_progress=clear_progress,
        )
