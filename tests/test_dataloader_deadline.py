"""Real subprocess regressions for deadlines outside the measured epoch loop."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from imgread_benchmark.dataloader import (
    BenchmarkRunError,
    DataLoaderConfig,
    snapshot_files,
)
from imgread_benchmark.dataloader import runner
from tests.dataloader_helpers import make_images


@pytest.mark.parametrize("phase", ["reader", "manifest", "request"])
@pytest.mark.parametrize("in_thread", [False, True])
def test_preflight_timeout_never_launches_consumer(
    tmp_path, monkeypatch, phase, in_thread
):
    path = make_images(tmp_path, 1)[0]
    manifest = snapshot_files([path] * (500 if phase == "request" else 1))
    original = runner.subprocess.Popen
    processes = []

    def launch(command, *args, **kwargs):
        assert command[-1] == "imgread_benchmark.dataloader._preflight"
        code = "import json,sys,time\n"
        if phase != "request":
            code += "json.loads(sys.stdin.readline())\n"
            code += f"print(json.dumps({{'kind':'phase','stage':'preflight.{phase}'}}), flush=True)\n"
        code += "time.sleep(60)\n"
        process = original([command[0], "-c", code], *args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "_SHUTDOWN_GRACE_SECONDS", 0.2)
    start = time.monotonic()

    def invoke():
        return runner.run_benchmark(manifest, DataLoaderConfig(), timeout_seconds=0.4)

    with pytest.raises(BenchmarkRunError) as caught:
        if in_thread:
            with ThreadPoolExecutor(1) as executor:
                executor.submit(invoke).result(timeout=5)
        else:
            invoke()
    assert time.monotonic() - start < 3
    assert "timeout" in caught.value.result.error["reason"]
    assert caught.value.result.error["stage"].startswith("preflight")
    assert not caught.value.result.consumer
    assert processes and all(
        p.poll() is not None and not runner.group_members(p.pid) for p in processes
    )


def test_preflight_and_consumer_share_one_budget(tmp_path, monkeypatch):
    original = runner.subprocess.Popen
    processes = []
    reader = dict(id="pil-rgb", description="test", available=True)

    def launch(command, *args, **kwargs):
        code = "import json,sys,time\njson.loads(sys.stdin.readline())\n"
        if command[-1].endswith("_preflight"):
            code += "time.sleep(0.3)\n"
            code += f"print({json.dumps(dict(kind='ready', reader=reader))!r}, flush=True)\n"
        else:
            code += "time.sleep(60)\n"
        process = original([command[0], "-c", code], *args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "_SHUTDOWN_GRACE_SECONDS", 0.2)
    start = time.monotonic()
    with pytest.raises(BenchmarkRunError) as caught:
        runner.run_benchmark(
            snapshot_files(make_images(tmp_path, 1)),
            DataLoaderConfig(),
            timeout_seconds=0.7,
        )
    assert 0.6 < time.monotonic() - start < 1.4
    assert caught.value.result.error["stage"] == "consumer preparation"
    assert len(processes) == 2
    assert all(not runner.group_members(p.pid) for p in processes)
