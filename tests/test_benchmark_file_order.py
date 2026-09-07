import builtins
from contextlib import contextmanager

import pytest

from imgread_benchmark import benchmark as benchmark_mod


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_shuffle_each_reader_repeat_preserves_sample_and_excludes_setup_time(
    monkeypatch, multiprocessing
):
    filenames = [f"img{i}.jpg" for i in range(5)]
    original = filenames[:]
    orders = []
    reads = []
    pool_sizes = []
    timing = False
    shuffle_calls = 0

    def shuffle(items):
        nonlocal shuffle_calls
        assert not timing
        shuffle_calls += 1
        items.reverse()

    def timeit(func, setup, number):
        nonlocal timing
        assert number == 1
        setup()
        orders.append(bench.item_list[:])
        timing = True
        try:
            func()
        finally:
            timing = False
        return 1.0

    class DummyPool:
        def __init__(self, num_workers):
            pool_sizes.append(num_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap(self, func, items):
            return map(func, items)

    monkeypatch.setattr(benchmark_mod.random, "shuffle", shuffle)
    monkeypatch.setattr(benchmark_mod, "timeit", timeit)
    monkeypatch.setattr("benchmark_utils.benchmark.Pool", DummyPool)
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=filenames,
        func_dict={"first": reads.append, "second": reads.append},
        num_repeats=3,
        shuffle=True,
        warmup=False,
        clear_progress=True,
    )
    bench.run(num_samples=3, multiprocessing=multiprocessing, num_workers=1)

    assert shuffle_calls == 6
    assert all(sorted(order) == original[:3] for order in orders)
    assert all(left != right for left, right in zip(orders, orders[1:]))
    assert reads == [filename for order in orders for filename in order]
    assert pool_sizes == ([1] * 6 if multiprocessing else [])
    assert bench._results == {"first": [1.0] * 3, "second": [1.0] * 3}
    assert not bench.exceptions
    assert filenames == original
    assert bench.item_list is filenames


@pytest.mark.parametrize("filenames", [[], ["only.jpg"], ["a.jpg", "b.jpg"]])
def test_shuffle_handles_small_samples_and_repeated_random_order(
    monkeypatch, filenames
):
    orders = []

    def timeit(func, setup, number):
        setup()
        orders.append(bench.item_list[:])
        func()
        return 1.0

    monkeypatch.setattr(benchmark_mod.random, "shuffle", lambda items: None)
    monkeypatch.setattr(benchmark_mod, "timeit", timeit)
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=filenames,
        func_dict={"dummy": lambda path: path},
        shuffle=True,
        warmup=False,
        num_repeats=3,
        clear_progress=True,
    )
    bench()

    assert all(sorted(order) == filenames for order in orders)
    if len(filenames) > 1:
        assert all(left != right for left, right in zip(orders, orders[1:]))


def test_file_order_is_stable_by_default():
    filenames = ["c.jpg", "a.jpg", "b.jpg"]
    reads = []
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=filenames,
        func_dict={"dummy": reads.append},
        num_repeats=3,
        warmup=False,
        clear_progress=True,
    )
    bench.run()
    assert reads == filenames * 3


@pytest.mark.parametrize("invoke", ["run", "__call__"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_warmup_reads_selected_files_once_before_timing(
    tmp_path, monkeypatch, invoke, shuffle
):
    filenames = [str(tmp_path / f"img{i}.jpg") for i in range(3)]
    contents = b"x" * (1024 * 1024 + 17)
    for filename in filenames:
        with builtins.open(filename, "wb") as file:
            file.write(contents)
    events = []

    @contextmanager
    def open_file(filename, mode):
        assert mode == "rb"
        with builtins.open(filename, mode) as file:

            class TrackedFile:
                def read(self, size):
                    assert 0 < size <= 1024 * 1024
                    chunk = file.read(size)
                    events.append(("warmup", filename, len(chunk)))
                    return chunk

            yield TrackedFile()

    def timeit(func, number, setup=lambda: None):
        setup()
        events.append(("timing",))
        func()
        return 1.0

    monkeypatch.setattr(benchmark_mod, "open", open_file, raising=False)
    monkeypatch.setattr(benchmark_mod, "timeit", timeit)
    monkeypatch.setattr("benchmark_utils.benchmark.timeit", timeit)
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=filenames,
        func_dict={"first": lambda path: None, "second": lambda path: None},
        num_repeats=2,
        shuffle=shuffle,
        clear_progress=True,
    )
    for _ in range(2):
        events.clear()
        getattr(bench, invoke)(num_samples=2)
        expected_warmup = [
            ("warmup", filename, size)
            for filename in filenames[:2]
            for size in (1024 * 1024, 17, 0)
        ]
        assert events == expected_warmup + [("timing",)] * 4


def test_no_warmup_when_no_readers_are_selected(monkeypatch):
    def fail_open(*args):
        pytest.fail("Warmup must not read files when there is nothing to benchmark")

    monkeypatch.setattr(benchmark_mod, "open", fail_open, raising=False)
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=["missing.jpg"], func_dict={"dummy": lambda path: None}
    )
    bench.run(func_name="unknown")
    assert not bench.results


def test_shuffle_restores_file_list_if_benchmark_fails(monkeypatch):
    filenames = ["a.jpg", "b.jpg"]

    def fail_timeit(func, setup, number):
        setup()
        raise RuntimeError("timer failed")

    monkeypatch.setattr(benchmark_mod, "timeit", fail_timeit)
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=filenames,
        func_dict={"dummy": lambda path: None},
        shuffle=True,
        warmup=False,
        clear_progress=True,
    )
    with pytest.raises(RuntimeError, match="timer failed"):
        bench.run()
    assert bench.item_list is filenames
    assert filenames == ["a.jpg", "b.jpg"]


@pytest.mark.parametrize("fail_on_open", [False, True])
def test_warmup_io_failure_reports_filename_before_timing(monkeypatch, fail_on_open):
    calls = []

    class UnreadableFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            raise OSError("read failed")

    def open_file(*args):
        if fail_on_open:
            raise PermissionError("access denied")
        return UnreadableFile()

    monkeypatch.setattr(benchmark_mod, "open", open_file, raising=False)
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=["unreadable.jpg"], func_dict={"dummy": calls.append}
    )
    with pytest.raises(benchmark_mod.FileWarmupError, match="unreadable.jpg") as exc:
        bench.run()
    assert isinstance(exc.value.__cause__, OSError)
    assert not calls
    assert not bench.results
