"""External coordinator: clean consumer per call and bounded process-group cleanup."""

from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from pathlib import Path

from .manifest import validate_manifest
from .models import BenchmarkResult, BenchmarkRunError, EpochResult, digest

_CLEANUP_FAILED = False


def group_members(group):
    """Linux process-group membership without requiring optional psutil."""
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                result.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


def cleanup_group(process, *, cancel=False):
    global _CLEANUP_FAILED
    if cancel and process.poll() is None:
        try:
            process.stdin.write(json.dumps({"event": "cancel"}) + "\n")
            process.stdin.flush()
        except (OSError, ValueError):
            pass
    for sig, seconds in ((None, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        if sig is not None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_members(process.pid):
                process.wait(timeout=1)
                return
            time.sleep(0.025)
    _CLEANUP_FAILED = True
    raise RuntimeError(
        f"Could not stop consumer process group {process.pid}; further runs disabled"
    )


def run_benchmark(manifest, config, *, timeout_seconds=None):
    invoked_at = time.monotonic()
    if _CLEANUP_FAILED:
        raise RuntimeError(
            "Previous process cleanup failed; refusing another configuration"
        )
    if not sys.platform.startswith("linux"):
        raise ValueError(
            "dataloader execution is qualified only on Linux (POSIX process groups)"
        )
    config.validate_count(manifest.selected_n)
    if timeout_seconds is not None and (
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive")
    from .readers import probe_reader

    reader = probe_reader(config.reader)
    if not reader.available:
        raise ValueError(f"reader={config.reader}: {reader.reason}")
    sampler_class = None
    if config.monitor_resources:
        from .resources import ResourceSampler, preflight_monitor

        preflight_monitor()
        sampler_class = ResourceSampler
    execution_id = str(uuid.uuid4())
    config_id = digest(
        dict(config=config.effective(), selection_id=manifest.selection_id)
    )
    result = BenchmarkResult(
        config_id,
        execution_id,
        manifest,
        {"requested": asdict(config), "effective": config.effective()},
        reader=asdict(reader),
        resource_status="partial" if config.monitor_resources else "off",
    )
    try:
        validate_manifest(manifest, config.reader)
    except ValueError as exc:
        result.error = dict(
            stage="preflight",
            reader=config.reader,
            path=getattr(exc, "path", None),
            reason=str(exc),
        )
        raise BenchmarkRunError(result) from exc
    events, stderr = queue.Queue(), deque(maxlen=1024)
    process = sampler = None
    readers = []
    done = False
    deadline = invoked_at + timeout_seconds if timeout_seconds else None
    old_sigterm = None
    if threading.current_thread() is threading.main_thread():
        old_sigterm = signal.getsignal(signal.SIGTERM)

        def cancel_on_sigterm(signum, frame):
            raise KeyboardInterrupt("SIGTERM received")

        signal.signal(signal.SIGTERM, cancel_on_sigterm)

    def drain(stream, kind):
        try:
            for line in stream:
                if kind == "stderr":
                    stderr.append(line)
                else:
                    events.put(line)
        finally:
            if kind == "stdout":
                events.put(None)

    try:
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "imgread_benchmark.dataloader._child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            cwd=".",
            env=environment,
        )
        for stream, kind in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            thread = threading.Thread(target=drain, args=(stream, kind), daemon=True)
            thread.start()
            readers.append(thread)
        process.stdin.write(
            json.dumps(
                dict(
                    protocol_version=1,
                    event="run_request",
                    config_id=config_id,
                    execution_id=execution_id,
                    manifest=manifest.to_dict(),
                    config=asdict(config),
                ),
                allow_nan=False,
            )
            + "\n"
        )
        process.stdin.flush()
        while True:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("configuration timeout")
            try:
                line = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break
            event = json.loads(line)
            if (
                event.get("protocol_version") != 1
                or event.get("config_id") != config_id
            ):
                raise RuntimeError("invalid child protocol identity")
            kind = event["event"]
            if kind == "ready":
                if result.consumer:
                    raise RuntimeError("duplicate ready")
                for key in ("reader", "environment", "consumer", "preparation"):
                    setattr(result, key, event[key])
                if sampler_class is not None:
                    sampler = sampler_class(
                        result.consumer,
                        config.sample_interval_ms / 1000,
                        config_id,
                        config.num_workers,
                    )
                    sampler.start()
                    result.baseline = sampler.baseline
                process.stdin.write(
                    json.dumps(dict(event="go", config_id=config_id)) + "\n"
                )
                process.stdin.flush()
            elif kind == "worker_started":
                if sampler is not None:
                    sampler.register(event)
            elif kind == "epoch_result":
                result.epochs.append(EpochResult(**event["epoch"]))
                result.pinning = event["pinning"]
            elif kind == "run_error":
                result.error = event["error"]
            elif kind == "done":
                done = True
                result.status = event["status"]
            else:
                raise RuntimeError(f"unknown child event {kind}")
        if not done or process.wait(timeout=10) != 0 or result.error:
            result.status = "failed"
            if result.error is None:
                raise RuntimeError("consumer EOF/nonzero exit without successful done")
        if result.status == "success" and (
            len(result.epochs) != config.epochs
            or any(e.status != "success" for e in result.epochs)
        ):
            raise RuntimeError("incomplete epoch sequence")
    except BaseException as exc:
        result.status = "failed"
        result.error = dict(
            stage="coordinator",
            reader=config.reader,
            path=None,
            reason=f"{type(exc).__name__}: {exc}",
            cancelled=isinstance(exc, KeyboardInterrupt),
        )
    finally:
        if sampler is not None:
            sampler.stop()
            sampler.apply(result)
        if process is not None:
            try:
                cleanup_group(process, cancel=not done)
            except Exception as exc:
                result.status = "failed"
                result.error = dict(
                    stage="cleanup", reader=config.reader, path=None, reason=str(exc)
                )
            for thread in readers:
                thread.join(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        if stderr:
            result.warnings.append("".join(stderr))
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
    if result.status != "success":
        raise BenchmarkRunError(result)
    return result
