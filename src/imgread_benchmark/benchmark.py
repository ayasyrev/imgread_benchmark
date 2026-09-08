from __future__ import annotations

import random
from timeit import timeit
from typing import Callable, Dict, List, Literal, Optional, Union

from benchmark_utils import BenchmarkIter
from rich.progress import track

from .get_img_filenames import get_img_filenames


def _get_read_to_format() -> Dict[str, Callable]:
    from .read_img import get_read_img, get_read_img_ndarray, get_read_img_pil

    return {
        "def": get_read_img(),
        "pil": get_read_img_pil(),
        "np": get_read_img_ndarray(),
    }


__all__ = ["BenchmarkImgRead", "FileWarmupError"]


class FileWarmupError(RuntimeError):
    """A selected file could not be read during cache warmup."""


class BenchmarkImgRead(BenchmarkIter):
    """Benchmark image reads with an optional shuffled order and file-cache warmup.

    Warmup reads the selected files once before each run, without decoding them.
    With shuffle enabled, each reader's repeat gets a fresh file order. Neither
    warmup nor shuffling contributes to the measured time.
    """

    def __init__(
        self,
        img_path: Optional[str] = None,
        num_samples: int = 0,
        target_format: Literal["def", "pil", "np"] = "def",
        func_dict: Optional[Dict[str, Callable]] = None,
        filenames: Optional[List[str]] = None,
        num_repeats: int = 5,
        clear_progress: bool = False,
        shuffle: bool = False,
        warmup: bool = True,
    ):
        self._target_format = target_format
        self.shuffle = shuffle
        self.warmup = warmup
        self._previous_order: Optional[List[str]] = None
        func_to_test = func_dict or _get_read_to_format()[target_format]
        img_path = img_path or "."
        if filenames is None:
            filenames = get_img_filenames(img_path, num_samples=num_samples)
        super().__init__(
            func=func_to_test,
            item_list=filenames,
            num_repeats=num_repeats,
            clear_progress=clear_progress,
        )

    def _run(
        self,
        func_names: Union[List[str], Dict[str, Callable]],
        num_repeats: Optional[int] = None,
    ) -> None:
        self._previous_order = None
        if func_names and self.warmup:
            filenames = self.item_list[: self._num_samples or len(self.item_list)]
            for filename in track(
                filenames,
                description="Warming up file cache",
                transient=self.clear_progress,
            ):
                try:
                    with open(filename, "rb") as file:
                        while file.read(1024 * 1024):
                            pass
                except OSError as exc:
                    raise FileWarmupError(
                        f"Failed to warm up '{filename}': {exc}"
                    ) from exc
        super()._run(func_names, num_repeats)

    def _run_benchmark(self, func_name: str, num_repeats: int) -> List[float]:
        if not self.shuffle:
            return super()._run_benchmark(func_name, num_repeats)

        # Shuffle only the selected sample, and never mutate the caller's list.
        original_items = self.item_list
        self.item_list = original_items[: self._num_samples or len(original_items)]

        def prepare_order() -> None:
            previous = self._previous_order or self.item_list[:]
            random.shuffle(self.item_list)
            if len(self.item_list) > 1 and self.item_list == previous:
                self.item_list.append(self.item_list.pop(0))
            self._previous_order = self.item_list[:]

        try:
            run_func = self.run_func_iter(func_name)
            name = f"[blue]{func_name:{self._max_name_len}}"
            task = self.progress_bar.add_task(name, total=num_repeats)
            run_times = []
            for repeat in range(num_repeats):
                self.progress_bar.update(
                    task, description=f"{name}: run {repeat + 1}/{num_repeats}"
                )
                run_times.append(timeit(run_func, setup=prepare_order, number=1))
                self.progress_bar.update(task, advance=1)
            average = sum(run_times) / len(run_times)
            self.progress_bar.update(
                task, description=f"{name}: {average:0.2f} sec/run."
            )
            return run_times
        finally:
            self.item_list = original_items

    @property
    def target_format(self) -> Literal["def", "pil", "np"]:
        return self._target_format

    @target_format.setter
    def target_format(self, target_format: Literal["def", "pil", "np"]) -> None:
        read_to_format = _get_read_to_format()
        if target_format not in read_to_format:
            print(f"{target_format} not in available format.")
            print("available: ", ", ".join(read_to_format.keys()))
        else:
            self._target_format = target_format
            self.func_dict = read_to_format[target_format]

    @property
    def func_names(self) -> str:
        return f"target_format: {self.target_format}, image libs: {', '.join(self.func_dict.keys())}"
