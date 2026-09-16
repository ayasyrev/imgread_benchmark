"""Decode prepared buffers, with explicit timing and supervised spawn workers."""

from __future__ import annotations

import multiprocessing
import hashlib
import json
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler

from PIL import Image
from rich import print as rprint

from .encoded_cache import prepare_cache, open_cache

START_TIMEOUT = 120
PROGRESS_TIMEOUT = 120
SHUTDOWN_GRACE = 10
SHUTDOWN_TERM = 5
SHUTDOWN_KILL = 2
_CLEANUP_FAILED = False


class DecodeRunError(RuntimeError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(
            "decode-only failed: "
            + "; ".join(f"{name}: {error}" for name, error in errors.items())
        )


class DecodeFailure(RuntimeError):
    def __init__(self, index, reason):
        self.index = index
        super().__init__(f"image index {index}: {reason}")


class DecodeCleanupError(RuntimeError):
    pass


def _decode_one(function, data):
    result = function(data)
    if result is None:
        raise ValueError("decoder returned None")
    if isinstance(result, Image.Image):
        result.load()
    del result


def _worker(connection, descriptor, function):
    try:
        with open_cache(descriptor) as lease:
            lease.prefault()
            _decode_one(function, lease.views[0])
            connection.send(("ready", os.getpid()))
            while True:
                task = connection.recv()
                if task is None:
                    return
                pass_id, index = task
                reason = None
                try:
                    _decode_one(function, lease.views[index])
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"[:2048]
                connection.send(("result", pass_id, index, reason))
    except (EOFError, BrokenPipeError):
        return
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"[:2048]))
        except (OSError, ValueError):
            pass
    finally:
        connection.close()


class DecodeWorkers:
    def __init__(self, descriptor, function, count):
        self.processes, self.connections = [], []
        context = multiprocessing.get_context("spawn")
        try:
            until = time.monotonic() + START_TIMEOUT
            for _ in range(count):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker, args=(child, descriptor, function)
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            pending = set(self.connections)
            while pending:
                connection, event = self.receive(pending, until, "worker startup")
                if event[0] != "ready":
                    raise RuntimeError(f"invalid worker readiness: {event}")
                pending.remove(connection)
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note(f"Worker cleanup failed: {cleanup}")
            raise

    def receive(self, connections, until, stage):
        while True:
            for process in self.processes:
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"worker {process.pid} exited during {stage} (exitcode={process.exitcode})"
                    )
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"decode-only timeout during {stage}")
            ready = wait(list(connections), timeout=min(0.1, remaining))
            if ready:
                connection = ready[0]
                try:
                    event = connection.recv()
                except EOFError as exc:
                    raise RuntimeError(f"worker EOF during {stage}") from exc
                if event[0] == "error":
                    raise RuntimeError(event[1])
                return connection, event

    def run_pass(self, order, pass_id):
        start = time.perf_counter()
        pending, next_index, completed = {}, 0, 0
        deadline = time.monotonic() + PROGRESS_TIMEOUT
        while completed < len(order):
            for connection in self.connections:
                if connection not in pending and next_index < len(order):
                    index = order[next_index]
                    connection.send((pass_id, index))
                    pending[connection] = index
                    next_index += 1
            connection, event = self.receive(pending, deadline, "decode pass")
            if len(event) != 4 or event[:3] != ("result", pass_id, pending[connection]):
                raise RuntimeError(f"invalid worker result: {event}")
            index = pending.pop(connection)
            if event[3] is not None:
                raise DecodeFailure(index, event[3])
            completed += 1
            deadline = time.monotonic() + PROGRESS_TIMEOUT
        return time.perf_counter() - start

    def close(self):
        global _CLEANUP_FAILED
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(None)
                except (OSError, ValueError):
                    pass
        for action, seconds in (
            (None, SHUTDOWN_GRACE),
            ("terminate", SHUTDOWN_TERM),
            ("kill", SHUTDOWN_KILL),
        ):
            if action:
                for process in self.processes:
                    if process.is_alive():
                        getattr(process, action)()
            until = time.monotonic() + seconds
            for process in self.processes:
                process.join(timeout=max(0, until - time.monotonic()))
            if not any(process.is_alive() for process in self.processes):
                break
        alive = [process.pid for process in self.processes if process.is_alive()]
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            if not process.is_alive():
                process.close()
        self.connections.clear()
        self.processes.clear()
        if alive:
            _CLEANUP_FAILED = True
            raise DecodeCleanupError(
                f"workers could not be stopped: {alive}; further decode runs disabled"
            )


def _sequential_pass(function, views, order, clock=time.perf_counter):
    start = clock()
    for index in order:
        try:
            _decode_one(function, views[index])
        except Exception as exc:
            raise DecodeFailure(index, f"{type(exc).__name__}: {exc}") from exc
    return clock() - start


def run_decode(
    functions,
    filenames,
    *,
    cache_dir=None,
    cache_limit=2 * 1024**3,
    num_repeats=5,
    shuffle=False,
    warmup=True,
    multiprocessing=False,
    num_workers=None,
):
    if _CLEANUP_FAILED:
        raise DecodeCleanupError(
            "previous worker cleanup failed; further decode runs disabled"
        )
    if not functions or not filenames:
        raise ValueError(
            "decode-only requires selected images and supported buffer decoders"
        )
    if type(num_repeats) is not int or num_repeats <= 0:
        raise ValueError("num_repeats must be positive")
    if num_workers is not None and (type(num_workers) is not int or num_workers < 0):
        raise ValueError("num_workers must be nonnegative")
    if multiprocessing:
        for name, function in functions.items():
            try:
                ForkingPickler.dumps(function)
            except Exception as exc:
                raise ValueError(
                    f"{name}: buffer decoder is not spawn-pickleable: {exc}"
                ) from exc
    for name, function in functions.items():
        if not callable(function):
            raise ValueError(f"{name}: buffer decoder is not callable")
    started = time.perf_counter()
    with prepare_cache(
        filenames, cache_dir=cache_dir, cache_limit=cache_limit
    ) as lease:
        lease.prefault()
        count = (num_workers or (os.cpu_count() or 1)) if multiprocessing else 0
        metadata = dict(
            mode=(
                "decode-only multiprocessing"
                if multiprocessing
                else "decode-only sequential"
            ),
            selected_n=len(filenames),
            cache_hit=lease.cache_hit,
            cache_path=str(lease.path),
            cache_id=lease.path.name,
            encoded_bytes=lease.encoded_bytes,
            selected_encoded_bytes=sum(length for _, length in lease.ranges),
            selection_id=hashlib.sha256(
                json.dumps(
                    [str(Path(path).expanduser().resolve()) for path in filenames],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            environment={
                "python": platform.python_version(),
                "imgread_benchmark": version("imgread_benchmark"),
                "numpy": version("numpy"),
                "pillow": version("pillow"),
                "start_method": "spawn" if multiprocessing else None,
            },
            evicted=lease.evicted,
            preparation_seconds=time.perf_counter() - started,
            warmup=warmup,
            shuffle=shuffle,
            workers=count,
            initialization_probe=True,
        )
        rprint(
            f"{metadata['mode']}: {len(filenames)} images, {lease.encoded_bytes} encoded bytes; cache {'hit' if lease.cache_hit else 'built'}: {lease.path}"
        )
        rprint(
            f"Preparation: {metadata['preparation_seconds']:.3f}s; warmup={warmup}, shuffle={shuffle}, workers={count}"
        )
        if multiprocessing:
            rprint(
                "Throughput includes task dispatch and result synchronization; worker startup is excluded."
            )
        if lease.evicted:
            rprint(f"Evicted caches: {lease.evicted}")
        results, errors = {}, {}
        for name, function in functions.items():
            pool = None
            stage = "initialization"
            try:
                if multiprocessing:
                    pool = DecodeWorkers(lease.descriptor, function, count)
                    run_pass = pool.run_pass
                else:
                    try:
                        _decode_one(function, lease.views[0])
                    except Exception as exc:
                        raise DecodeFailure(0, str(exc)) from exc

                    def run_pass(order, pass_id):
                        return _sequential_pass(function, lease.views, order)

                order = list(range(len(filenames)))
                if warmup:
                    stage = "warmup"
                    run_pass(order, -1)
                durations = []
                for repeat in range(num_repeats):
                    stage = f"repeat {repeat + 1}"
                    if shuffle:
                        random.shuffle(order)
                    durations.append(run_pass(order, repeat))
                    rprint(
                        f"{name}: repeat {repeat + 1}/{num_repeats} completed in {durations[-1]:.4f}s"
                    )
                results[name] = durations
            except Exception as exc:
                index = getattr(exc, "index", None)
                errors[name] = dict(
                    stage=stage,
                    reason=f"{type(exc).__name__}: {exc}",
                    index=index,
                    path=str(filenames[index]) if index is not None else None,
                )
            finally:
                if pool is not None:
                    try:
                        pool.close()
                    except Exception as exc:
                        results.pop(name, None)
                        error = errors.setdefault(
                            name, dict(stage="cleanup", reason=str(exc))
                        )
                        error.setdefault("cleanup_errors", []).append(str(exc))
            if _CLEANUP_FAILED:
                break
        return results, errors, metadata
