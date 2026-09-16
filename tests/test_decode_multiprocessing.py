import multiprocessing
from pathlib import Path
import tempfile
import time

import pytest

from imgread_benchmark import decode_benchmark as decode
from imgread_benchmark import encoded_cache as cache
from tests.decode_helpers import (
    record_decode,
    crash_decode,
    hang_decode,
    invalid_decode,
    hold_lease,
)


@pytest.fixture
def real_cache():
    try:
        directory = tempfile.TemporaryDirectory(prefix="imgread-test-", dir="/dev/shm")
    except (PermissionError, FileNotFoundError):
        pytest.skip("requires writable Linux tmpfs")
    with directory:
        yield Path(directory.name)


@pytest.fixture
def sources(tmp_path):
    paths = [tmp_path / f"{i}.bin" for i in range(3)]
    for i, path in enumerate(paths):
        path.write_bytes(bytes([i]))
    return paths


def test_spawn_counts_and_reuse(sources, real_cache, tmp_path, monkeypatch):
    log = tmp_path / "calls"
    monkeypatch.setenv("IMGREAD_TEST_DECODE_LOG", str(log))
    before = {process.pid for process in multiprocessing.active_children()}
    results, errors, report = decode.run_decode(
        {"record": record_decode},
        sources,
        cache_dir=real_cache,
        multiprocessing=True,
        num_workers=2,
        num_repeats=2,
        shuffle=True,
    )
    assert not errors and len(results["record"]) == 2
    calls = log.read_bytes()
    assert (
        len(calls) == 2 + 3 * 3
    )  # independent init probe per worker, warmup, two repeats
    assert calls.count(b"\0") == 5 and calls.count(b"\1") == calls.count(b"\2") == 3
    assert report["workers"] == 2
    assert {process.pid for process in multiprocessing.active_children()} == before
    with cache.prepare_cache(sources[:1], cache_dir=real_cache) as lease:
        assert lease.cache_hit


@pytest.mark.parametrize("function", [crash_decode, hang_decode, invalid_decode])
def test_failure_reaps_workers(sources, real_cache, monkeypatch, function):
    monkeypatch.setattr(decode, "PROGRESS_TIMEOUT", 0.5)
    monkeypatch.setattr(decode, "SHUTDOWN_GRACE", 0.1)
    before = {process.pid for process in multiprocessing.active_children()}
    started = time.monotonic()
    results, errors, _ = decode.run_decode(
        {"bad": function},
        sources,
        cache_dir=real_cache,
        multiprocessing=True,
        num_workers=2,
        num_repeats=1,
    )
    assert not results and "bad" in errors
    assert time.monotonic() - started < 15
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_worker_lease_survives_parent_close(sources, real_cache):
    context = multiprocessing.get_context("spawn")
    with cache.prepare_cache(sources[:1], cache_dir=real_cache) as lease:
        parent, child = context.Pipe()
        process = context.Process(target=hold_lease, args=(lease.descriptor, child))
        process.start()
        child.close()
        assert parent.poll(10)
        assert parent.recv() == b"\0"
    try:
        with pytest.raises(cache.EncodedCacheError, match="active caches"):
            cache.prepare_cache(sources[1:2], cache_dir=real_cache, cache_limit=8192)
    finally:
        parent.send(None)
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(3)
        parent.close()
    assert process.exitcode == 0
    process.close()
    with cache.prepare_cache(sources[1:2], cache_dir=real_cache, cache_limit=8192):
        pass
