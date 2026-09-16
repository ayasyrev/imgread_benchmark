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
        *,
        decode_only: bool = False,
        cache_dir: Optional[str] = None,
        cache_limit: Union[int, str] = 2147483648,
    ):
        self._target_format = target_format
        self.shuffle = shuffle
        self.warmup = warmup
        self._previous_order: Optional[List[str]] = None
        self.decode_only = decode_only
        self.cache_dir = cache_dir
        self.cache_limit = cache_limit
        self.decode_report = {}
        self.unsupported_decoders = {}
        if decode_only:
            from .encoded_cache import parse_cache_limit
            from .read_img import get_decode_functions

            self.cache_limit = parse_cache_limit(cache_limit)
            registry, self.unsupported_decoders = get_decode_functions(target_format)
            func_to_test = registry if func_dict is None else func_dict
            self._run = self._run_decode
            self.run = self._run_decode_selected
        else:
            if cache_dir is not None or cache_limit != 2147483648:
                raise ValueError("cache_dir/cache_limit require decode_only=True")
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

    def _run_decode_selected(
        self,
        func_name=None,
        exclude=None,
        num_repeats=None,
        num_samples=None,
        multiprocessing=None,
        num_workers=None,
    ):
        names = [func_name] if isinstance(func_name, str) else func_name
        excluded = [exclude] if isinstance(exclude, str) else (exclude or [])
        if names is not None:
            for name in names:
                if name not in self.func_dict:
                    reason = self.unsupported_decoders.get(
                        name, "unknown or unavailable reader"
                    )
                    raise ValueError(f"{name}: {reason}")
        selected = [
            name
            for name in (names if names is not None else self.func_dict)
            if name not in excluded
        ]
        self._num_samples, self._multiprocessing, self._num_workers = (
            num_samples,
            multiprocessing,
            num_workers,
        )
        try:
            self._run_decode(selected, num_repeats)
        finally:
            self._num_samples = self._multiprocessing = self._num_workers = None

    def _run_decode(self, func_names, num_repeats=None):
        from .decode_benchmark import DecodeRunError, run_decode
        from .read_img import get_read_img_version

        self._reset_results()
        self.decode_report = {}
        if self._num_samples is not None and (
            type(self._num_samples) is not int or self._num_samples < 0
        ):
            raise ValueError("num_samples must be nonnegative")
        filenames = self.item_list[: self._num_samples or len(self.item_list)]
        versions = get_read_img_version()
        for name in func_names:
            print(f"Decoder: {name} {versions.get(name, 'custom/unknown')}")
        for name, reason in self.unsupported_decoders.items():
            if name not in self.func_dict:
                print(f"Skipped: {name}: {reason}")
        self._results, errors, metadata = run_decode(
            {name: self.func_dict[name] for name in func_names},
            filenames,
            cache_dir=self.cache_dir,
            cache_limit=self.cache_limit,
            num_repeats=self.num_repeats if num_repeats is None else num_repeats,
            shuffle=self.shuffle,
            warmup=self.warmup,
            multiprocessing=bool(self._multiprocessing),
            num_workers=self._num_workers,
        )
        self.decode_report = {
            **metadata,
            "target_format": self.target_format,
            "versions": {
                name: versions.get(name, "custom/unknown") for name in func_names
            },
            "errors": errors,
            "seconds": self._results,
        }
        if self._results:
            self.print_results_per_item()
        if errors:
            raise DecodeRunError(errors)

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
        if self.decode_only:
            from .read_img import get_decode_functions

            self.func_dict, self.unsupported_decoders = get_decode_functions(
                target_format
            )
            self._target_format = target_format
            return
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
