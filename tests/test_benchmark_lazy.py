from functools import partial
from multiprocessing.reduction import ForkingPickler

from benchmark_utils.benchmark import try_run

from imgread_benchmark.read_img import graceful_degradation


def _dummy_reader_for_pickle(_path: str) -> str:
    return "ok"


def test_graceful_degradation_wrapper_is_pickleable_for_multiprocessing():
    wrapped = graceful_degradation(_dummy_reader_for_pickle)
    payload = partial(try_run, wrapped)
    ForkingPickler.dumps(payload)
