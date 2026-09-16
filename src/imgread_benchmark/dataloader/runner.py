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
from dataclasses import asdict
from pathlib import Path

from ._transport import Deadline, JsonTransport
from .models import BenchmarkResult, BenchmarkRunError, EpochResult, digest

_CLEANUP_FAILED = False
_SHUTDOWN_GRACE_SECONDS = 10
_SHUTDOWN_TERM_SECONDS = 5
_SHUTDOWN_KILL_SECONDS = 2


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


def cleanup_group(process, *, cancel=False, graceful_deadline=None):
    global _CLEANUP_FAILED
    if graceful_deadline is None:
        graceful_deadline = time.monotonic() + _SHUTDOWN_GRACE_SECONDS
    if cancel and process.poll() is None:
        try:
            # A full request pipe must never block cancellation.
            os.kill(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    for sig, seconds in (
        (None, None),
        (signal.SIGTERM, _SHUTDOWN_TERM_SECONDS),
        (signal.SIGKILL, _SHUTDOWN_KILL_SECONDS),
    ):
        if sig is not None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        deadline = graceful_deadline if sig is None else time.monotonic() + seconds
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


def _environment():
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def _run_preflight(manifest, config, deadline, result):
    global _CLEANUP_FAILED
    deadline.remaining("preflight")
    process = subprocess.Popen(
        [sys.executable, "-m", "imgread_benchmark.dataloader._preflight"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=_environment(),
    )
    transport = JsonTransport(process)
    stage, response, finished = "preflight", None, False
    # Retain a finite probe limit even for runs without a total budget.
    probe_deadline = Deadline(
        min(deadline.expires or float("inf"), time.monotonic() + 30)
    )
    try:
        transport.send_factory(
            lambda: dict(manifest=manifest.to_dict(), config=asdict(config))
        )
        while True:
            try:
                line = transport.receive(probe_deadline, stage)
            except queue.Empty:
                continue
            if line is None:
                break
            event = json.loads(line)
            if event["kind"] == "phase":
                stage = event["stage"]
            else:
                response = event
        process.wait(timeout=probe_deadline.remaining(stage))
        finished = True
        if response is None:
            raise RuntimeError("preflight exited without a result")
        if response["kind"] == "configuration_error":
            raise ValueError(response["reason"])
        if response["kind"] == "error":
            result.error = response["error"]
            raise BenchmarkRunError(result)
        if response["kind"] != "ready" or process.returncode:
            raise RuntimeError("invalid preflight result")
        result.reader = response["reader"]
        deadline.remaining(stage)
    except BaseException as exc:
        if result.error is None:
            result.error = dict(
                stage=getattr(exc, "stage", stage),
                reader=config.reader,
                path=None,
                reason=f"{type(exc).__name__}: {exc}",
                cancelled=isinstance(exc, KeyboardInterrupt),
            )
        raise
    finally:
        try:
            cleanup_group(process, cancel=not finished)
            transport.close()
        except Exception as exc:
            _CLEANUP_FAILED = True
            if result.error is None:
                raise
            result.error.setdefault("secondary_errors", []).append(
                dict(stage="cleanup", reason=str(exc))
            )
        if transport.stderr:
            result.warnings.append("".join(transport.stderr))


def run_benchmark(manifest, config, *, timeout_seconds=None):
    global _CLEANUP_FAILED
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
    execution_id = str(uuid.uuid4())
    config_id = digest(
        dict(config=config.effective(), selection_id=manifest.selection_id)
    )
    result = BenchmarkResult(
        config_id,
        execution_id,
        manifest,
        {"requested": asdict(config), "effective": config.effective()},
        resource_status="partial" if config.monitor_resources else "off",
    )
    process = sampler = transport = None
    done = False
    deadline = Deadline(invoked_at + timeout_seconds if timeout_seconds else None)
    teardown_deadline = None
    old_sigterm = None
    stage = "preflight"
    configuration_error = None

    def record_error(error):
        if result.error is None:
            result.error = error
        elif result.error != error:
            result.error.setdefault("secondary_errors", []).append(error)
            if error.get("cancelled"):
                result.error["cancelled"] = True

    if threading.current_thread() is threading.main_thread():
        old_sigterm = signal.getsignal(signal.SIGTERM)

        def cancel_on_sigterm(signum, frame):
            raise KeyboardInterrupt("SIGTERM received")

        signal.signal(signal.SIGTERM, cancel_on_sigterm)

    try:
        try:
            _run_preflight(manifest, config, deadline, result)
        except ValueError as exc:
            configuration_error = exc
            raise
        stage = "consumer startup"
        deadline.remaining(stage)
        process = subprocess.Popen(
            [sys.executable, "-m", "imgread_benchmark.dataloader._child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            cwd=".",
            env=_environment(),
        )
        transport = JsonTransport(process)
        transport.send_factory(
            lambda: dict(
                protocol_version=1,
                event="run_request",
                config_id=config_id,
                execution_id=execution_id,
                manifest=manifest.to_dict(),
                config=asdict(config),
            )
        )
        stage = "consumer preparation"
        while True:
            deadline.remaining(stage)
            if teardown_deadline is not None and time.monotonic() > teardown_deadline:
                raise TimeoutError("consumer shutdown timeout while waiting for EOF")
            try:
                line = transport.receive(deadline, stage)
            except queue.Empty:
                continue
            if line is None:
                if teardown_deadline is None:
                    teardown_deadline = time.monotonic() + _SHUTDOWN_GRACE_SECONDS
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
                if config.monitor_resources:
                    stage = "monitor startup"
                    from .resources import ResourceSampler

                    sampler = ResourceSampler(
                        result.consumer,
                        config.sample_interval_ms / 1000,
                        config_id,
                        config.num_workers,
                    )
                    sampler.start(timeout=deadline.remaining("monitor startup"))
                    result.baseline = sampler.baseline
                deadline.remaining("epoch startup")
                transport.send(dict(event="go", config_id=config_id))
                stage = "epochs"
            elif kind == "worker_started":
                if sampler is not None:
                    sampler.register(event)
            elif kind == "epoch_result":
                result.epochs.append(EpochResult(**event["epoch"]))
                result.pinning = event["pinning"]
                if teardown_deadline is None and (
                    result.epochs[-1].status != "success"
                    or len(result.epochs) >= config.epochs
                ):
                    teardown_deadline = time.monotonic() + _SHUTDOWN_GRACE_SECONDS
            elif kind == "run_error":
                record_error(event["error"])
                if teardown_deadline is None:
                    teardown_deadline = time.monotonic() + _SHUTDOWN_GRACE_SECONDS
            elif kind == "done":
                done = True
                result.status = event["status"]
                if teardown_deadline is None:
                    teardown_deadline = time.monotonic() + _SHUTDOWN_GRACE_SECONDS
            else:
                raise RuntimeError(f"unknown child event {kind}")
        # EOF can precede process exit. It shares the same grace budget as
        # terminal events and process-group cleanup, rather than resetting it.
        exit_deadline = (
            min(deadline.expires, teardown_deadline)
            if deadline.expires
            else teardown_deadline
        )
        try:
            returncode = process.wait(timeout=max(0, exit_deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("consumer shutdown timeout after EOF") from exc
        if not done or returncode != 0 or result.error:
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
        record_error(
            dict(
                stage=getattr(exc, "stage", stage),
                reader=config.reader,
                path=None,
                reason=f"{type(exc).__name__}: {exc}",
                cancelled=isinstance(exc, KeyboardInterrupt),
            )
        )
    finally:
        if sampler is not None:
            try:
                sampler.stop()
                sampler.apply(result)
            except Exception as exc:
                result.status = "failed"
                _CLEANUP_FAILED = True
                record_error(dict(stage="monitor cleanup", reason=str(exc)))
        if process is not None:
            process_clean = True
            try:
                cleanup_group(
                    process, cancel=not done, graceful_deadline=teardown_deadline
                )
            except Exception as exc:
                process_clean = False
                result.status = "failed"
                record_error(
                    dict(
                        stage="cleanup",
                        reader=config.reader,
                        path=None,
                        reason=str(exc),
                    )
                )
            if transport is not None and process_clean:
                try:
                    transport.close()
                except Exception as exc:
                    _CLEANUP_FAILED = True
                    record_error(dict(stage="cleanup", reason=str(exc)))
                    result.status = "failed"
                if transport.stderr:
                    result.warnings.append("".join(transport.stderr))
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
    if configuration_error is not None:
        raise configuration_error
    if result.status != "success":
        raise BenchmarkRunError(result)
    return result
