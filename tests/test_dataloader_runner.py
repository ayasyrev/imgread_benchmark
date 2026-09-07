import os

import pytest

pytest.importorskip("torch")

from imgread_benchmark.dataloader import (
    BenchmarkRunError,
    DataLoaderConfig,
    run_benchmark,
    snapshot_files,
)
from imgread_benchmark.dataloader.engine import consume_epoch
from imgread_benchmark.dataloader.runner import group_members
from tests.dataloader_helpers import make_images


def test_timer_boundaries():
    from imgread_benchmark.dataloader import EpochSampler
    import torch

    clock_value = [100]
    events = []

    class Loader:
        def __iter__(self):
            events.append("iter")
            clock_value[0] += 2
            yield torch.zeros((4, 3, 2, 2), dtype=torch.uint8), 0
            clock_value[0] += 3
            yield torch.zeros((2, 3, 2, 2), dtype=torch.uint8), 0
            clock_value[0] += 5
            events.append("exhausted")

    def clock():
        events.append("clock")
        return clock_value[0] * 10**9

    e, _, observed, error = consume_epoch(
        Loader(), EpochSampler(6, 0, False), 0, 6, DataLoaderConfig(), clock=clock
    )
    assert events == ["clock", "iter", "exhausted", "clock"]
    assert (
        e.epoch_seconds == 10 and e.images_delivered == 6 and e.batches_delivered == 2
    )
    assert error is None and observed is False


@pytest.mark.parametrize("workers,persistent", [(0, False), (2, False), (2, True)])
@pytest.mark.parametrize("monitor", [False, True])
def test_real_runner(tmp_path, workers, persistent, monitor):
    manifest = snapshot_files(make_images(tmp_path))
    config = DataLoaderConfig(
        num_workers=workers,
        persistent_workers=persistent,
        monitor_resources=monitor,
        epochs=3,
        batch_size=4,
        shuffle=True,
    )
    result = run_benchmark(manifest, config, timeout_seconds=60)
    assert result.status == "success"
    assert len(result.epochs) == 3
    assert all(
        e.images_delivered == 10 and e.batches_delivered == 3 for e in result.epochs
    )
    assert result.consumer["pid"] != os.getpid()
    assert not group_members(result.consumer["pid"])
    assert result.pinning["status"] == "not_requested"
    assert (
        result.late_epoch_seconds_median
        == sum(e.epoch_seconds for e in result.epochs[1:]) / 2
    )
    assert bool(result.baseline) == monitor
    if monitor:
        registrations = result.preparation["worker_registrations"]
        assert len(registrations) == (workers if persistent else workers * 3)
        assert result.resource_status in ("complete", "partial")
        for e in result.epochs:
            assert 0 <= e.resources["cpu_time_coverage"] <= 1
    else:
        assert result.resource_samples == []
        assert "worker_registrations" not in result.preparation
        assert result.environment["psutil"] is None


def test_drop_last_and_fresh_consumers(tmp_path):
    manifest = snapshot_files(make_images(tmp_path))
    config = DataLoaderConfig(epochs=1, batch_size=4, drop_last=True)
    results = [run_benchmark(manifest, config) for _ in range(2)]
    assert results[0].execution_id != results[1].execution_id
    assert results[0].consumer["pid"] != results[1].consumer["pid"]
    assert results[0].config_id == results[1].config_id
    for result in results:
        assert result.epochs[0].images_delivered == 8
        assert result.epochs[0].batches_delivered == 2
        assert result.epochs[0].images_dropped == 2
        assert result.late_epoch_seconds_median is None


def test_failure_and_cleanup(tmp_path):
    from PIL import Image

    paths = make_images(tmp_path, 2)
    Image.new("RGB", (17, 19)).save(paths[1])
    manifest = snapshot_files(paths)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmark(
            manifest, DataLoaderConfig(num_workers=2, geometry=False, batch_size=2)
        )
    result = caught.value.result
    assert "shapes" in result.error["reason"]
    assert result.epochs[0].images_per_second is None
    assert not group_members(result.consumer["pid"])


def test_timeout_cleanup(tmp_path):
    manifest = snapshot_files(make_images(tmp_path))
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmark(
            manifest, DataLoaderConfig(num_workers=2, epochs=1000), timeout_seconds=2
        )
    result = caught.value.result
    assert "timeout" in result.error["reason"]
    if result.consumer:
        assert not group_members(result.consumer["pid"])


def test_pin_requested(tmp_path):
    result = run_benchmark(
        snapshot_files(make_images(tmp_path, 1)),
        DataLoaderConfig(epochs=1, pin_memory=True),
    )
    assert result.pinning["requested"]
    assert result.pinning["status"] in ("observed_pinned", "observed_unpinned")


def test_failed_epoch_does_not_get_rates():
    import torch
    from imgread_benchmark.dataloader import EpochSampler

    class Broken:
        def __iter__(self):
            yield torch.zeros((4, 3, 2, 2), dtype=torch.uint8), 0
            raise RuntimeError("injected next failure")

    e, _, _, error = consume_epoch(
        Broken(), EpochSampler(10, 0, False), 0, 10, DataLoaderConfig()
    )
    assert e.status == "failed" and e.images_delivered == 4
    assert e.images_per_second is None and e.ms_per_image is None
    assert isinstance(error, RuntimeError)


def test_monitor_off_does_not_import_psutil(tmp_path, monkeypatch):
    import builtins
    from imgread_benchmark.dataloader import resources

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] == "psutil":
            raise ImportError("psutil blocked")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(
        resources, "ResourceSampler", lambda *args: pytest.fail("sampler created")
    )
    manifest = snapshot_files(make_images(tmp_path, 1))
    assert run_benchmark(manifest, DataLoaderConfig(epochs=1)).resource_status == "off"
    with pytest.raises(ValueError, match="psutil blocked"):
        run_benchmark(manifest, DataLoaderConfig(epochs=1, monitor_resources=True))


def test_corrupt_payload_in_worker(tmp_path):
    import struct

    paths = make_images(tmp_path, 1)
    data = bytearray(paths[0].read_bytes())
    offset = data.index(b"IDAT")
    size = struct.unpack(">I", data[offset - 4 : offset])[0]
    data[offset + 4 : offset + 4 + size] = b"x" * size
    paths[0].write_bytes(data)
    manifest = snapshot_files(paths)
    with pytest.raises(BenchmarkRunError) as caught:
        run_benchmark(manifest, DataLoaderConfig(num_workers=2, epochs=1))
    assert str(paths[0]) in caught.value.result.error["reason"]
    assert not group_members(caught.value.result.consumer["pid"])


def test_sigterm_and_forced_worker_death(tmp_path):
    import json
    import signal
    import subprocess
    import sys
    import time

    psutil = pytest.importorskip("psutil")
    # Exercise cancellation in a separate caller so pytest itself keeps its signal handlers.
    make_images(tmp_path, 10, shape=(320, 480))
    script = """
import json, sys
from pathlib import Path
from imgread_benchmark.dataloader import *
root = Path(sys.argv[1])
try:
    run_benchmark(snapshot_files([root / f'{i:03}.png' for i in range(10)]), DataLoaderConfig(num_workers=2, persistent_workers=True, epochs=10000, batch_size=4))
except BenchmarkRunError as exc:
    (root / 'failure.json').write_text(json.dumps(exc.result.to_dict()))
    raise SystemExit(130 if exc.result.error.get('cancelled') else 1)
"""
    for mode in ("sigterm", "worker_death", "worker_hang"):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        children = []
        try:
            deadline = time.monotonic() + 30
            workers = []
            while time.monotonic() < deadline:
                children = psutil.Process(process.pid).children(recursive=True)
                workers = [p for p in children if "spawn_main" in " ".join(p.cmdline())]
                if len(workers) == 2:
                    break
                if process.poll() is not None:
                    pytest.fail("test caller exited before worker startup")
                time.sleep(0.05)
            assert len(workers) == 2
            if mode in ("sigterm", "worker_hang"):
                if mode == "worker_hang":
                    workers[0].send_signal(signal.SIGSTOP)
                process.send_signal(signal.SIGTERM)
            else:
                workers[0].kill()
            assert process.wait(timeout=35) == (
                130 if mode in ("sigterm", "worker_hang") else 1
            )
            result = json.loads((tmp_path / "failure.json").read_text())
            assert result["status"] == "failed"
            assert not group_members(result["consumer"]["pid"])
            assert all(
                not p.is_running() or p.status() == psutil.STATUS_ZOMBIE
                for p in children
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass


@pytest.mark.parametrize("workers", [0, 2])
def test_real_monitor_slow_dataset(tmp_path, monkeypatch, workers):
    import subprocess

    original = subprocess.Popen

    def launch(command, *args, **kwargs):
        if command[1:] == ["-m", "imgread_benchmark.dataloader._child"]:
            command = [command[0], "-m", "tests.dataloader_helpers", "slow-child"]
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    config = DataLoaderConfig(
        num_workers=workers,
        monitor_resources=True,
        epochs=1,
        batch_size=2,
        sample_interval_ms=50,
    )
    result = run_benchmark(snapshot_files(make_images(tmp_path, 10)), config)
    resources = result.epochs[0].resources
    assert resources["sample_count"] >= 2
    assert resources["complete_process_interval_count"] >= 1
    assert resources["cpu_mean_percent"] is not None
    assert resources["rss"]["total"]["observed_peak_mib"] > 0
    observed = {p["role"] for s in result.resource_samples for p in s["processes"]}
    assert "consumer" in observed
    if workers:
        assert "worker" in observed
    assert not group_members(result.consumer["pid"])


def test_monitored_abba_fresh_processes(tmp_path):
    manifest = snapshot_files(make_images(tmp_path, 2))
    a = DataLoaderConfig(epochs=1, monitor_resources=True)
    b = DataLoaderConfig(
        reader="cv2-bgr-cvtcolor",
        num_workers=2,
        persistent_workers=True,
        epochs=1,
        monitor_resources=True,
    )
    results = [run_benchmark(manifest, config) for config in (a, b, b, a)]
    identities = {(r.consumer["pid"], r.consumer["os_create_time"]) for r in results}
    assert len(identities) == 4
    assert len({r.execution_id for r in results}) == 4
    assert len({r.config_id for r in results}) == 2
    for result in results:
        assert (
            result.baseline
            and result.baseline["sweep_end_ns"] <= result.epochs[0].start_ns
        )
        assert not group_members(result.consumer["pid"])
