"""Measured consumer loop. Only this module touches private torch teardown APIs."""

from __future__ import annotations

import os
import platform
import queue
import threading
import time
from dataclasses import asdict
from functools import partial
from importlib.metadata import version

from .models import EpochResult, digest


def thread_policy(reader):
    import torch

    torch.set_num_threads(1)
    # Interop can only be set before parallel work starts in a process.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    settings = {
        "torch_intra": torch.get_num_threads(),
        "torch_interop": torch.get_num_interop_threads(),
        "pillow": None,
        "pillow_reason": "no thread control API",
    }
    if reader.startswith("cv2"):
        import cv2

        cv2.setNumThreads(1)
        settings["opencv"] = cv2.getNumThreads()
    return settings


def worker_init(worker_id, *, reader, registrations):
    thread_policy(reader)
    if registrations is not None:
        try:
            import psutil

            registration = dict(
                worker_id=worker_id,
                pid=os.getpid(),
                create_time=psutil.Process().create_time(),
                registered_monotonic_ns=time.perf_counter_ns(),
            )
        except Exception as exc:
            registration = dict(
                worker_id=worker_id,
                pid=os.getpid(),
                create_time=None,
                registered_monotonic_ns=time.perf_counter_ns(),
                error=str(exc),
            )
        registrations.put(registration)


def byte_warmup(manifest):
    for path in manifest.paths:
        with open(path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass


def close_loader(loader, iterator):
    """Idempotent adapter pinned to torch 2.10.0, including nonpersistent iterators."""
    try:
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
    finally:
        if loader is not None:
            loader._iterator = None


def consume_epoch(loader, sampler, index, count, config, *, clock=time.perf_counter_ns):
    sampler.set_epoch(index)
    images = batches = 0
    batch = None
    iterator = None
    error = None
    start = clock()
    try:
        iterator = iter(loader)
        for batch in iterator:
            images += batch[0].shape[0]
            batches += 1
        end = clock()
    except BaseException as exc:
        end = clock()
        error = exc
    # All probes, hashes and reporting are outside the measured window.
    observed = bool(batch[0].is_pinned()) if batch is not None else None
    del batch
    order = list(sampler)
    dropped = count % config.batch_size if config.drop_last else 0
    epoch = EpochResult.measured(
        index,
        start,
        end,
        images,
        batches,
        dropped,
        digest(order),
        digest(order[:images]),
        status="failed" if error else "success",
    )
    return epoch, iterator, observed, error


def execute(manifest, config, execution_id, emit, wait_go):
    preparation_start = time.perf_counter_ns()
    started_wall_time = time.time()
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from .dataset import ContextDataset, EpochSampler, ImageListDataset, collate_images
    from .readers import check_runtime

    reader = check_runtime(config.reader)
    settings = thread_policy(config.reader)
    config.validate_count(manifest.selected_n)
    if config.warmup:
        byte_warmup(manifest)
    sampler = EpochSampler(manifest.selected_n, config.seed, config.shuffle)
    registrations = relay = None
    relay_stop = threading.Event()
    if config.monitor_resources and config.num_workers:
        import multiprocessing

        registrations = multiprocessing.get_context("spawn").Queue()

        def relay_events():
            while not relay_stop.is_set():
                try:
                    value = registrations.get(timeout=0.05)
                except queue.Empty:
                    continue
                emit("worker_started", **value)

        relay = threading.Thread(target=relay_events, daemon=True)
        relay.start()
    kwargs = {}
    if config.num_workers:
        kwargs.update(
            multiprocessing_context="spawn",
            prefetch_factor=config.effective_prefetch_factor,
        )
    loader = iterator = None
    try:
        loader = DataLoader(
            ContextDataset(ImageListDataset(manifest, config.reader, config.geometry)),
            sampler=sampler,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            collate_fn=collate_images,
            worker_init_fn=partial(
                worker_init, reader=config.reader, registrations=registrations
            ),
            generator=torch.Generator().manual_seed(config.seed),
            in_order=True,
            timeout=0,
            **kwargs,
        )
        environment = dict(
            python=platform.python_version(),
            package=version("imgread_benchmark"),
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            backend=reader.version,
            psutil=version("psutil") if config.monitor_resources else None,
            os=platform.system(),
            kernel=platform.release(),
            machine=platform.machine(),
            logical_cpus=os.cpu_count(),
            affinity=sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            start_method="spawn",
            threads=settings,
            thread_env={
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            geometry="uint8 CPU RGB; resize short=256 bilinear antialias=True; center_crop 224 ties-to-even"
            if config.geometry
            else "native size uint8 CPU RGB",
            torchvision_cpu_image_ops=hasattr(torch.ops.image, "decode_image"),
        )
        environment["available_cpus"] = (
            len(environment["affinity"])
            if environment["affinity"] is not None
            else None
        )
        consumer = dict(
            pid=os.getpid(),
            execution_id=execution_id,
            started_wall_time=started_wall_time,
            os_create_time=None,
            os_create_time_reason="monitoring off",
        )
        if config.monitor_resources:
            import psutil

            consumer.update(
                os_create_time=psutil.Process().create_time(),
                os_create_time_reason=None,
            )
        emit(
            "ready",
            reader=asdict(reader),
            environment=environment,
            consumer=consumer,
            preparation=dict(
                seconds=(time.perf_counter_ns() - preparation_start) / 1e9,
                warmup=config.warmup,
            ),
        )
        wait_go()
        for index in range(config.epochs):
            epoch, iterator, observed, error = consume_epoch(
                loader, sampler, index, manifest.selected_n, config
            )
            accelerator = torch.accelerator.is_available()
            pinning = dict(
                requested=config.pin_memory,
                last_batch_observed=observed,
                accelerator_available=accelerator,
                status="not_requested"
                if not config.pin_memory
                else "not_observed"
                if observed is None
                else "observed_pinned"
                if observed
                else "observed_unpinned",
                warnings=["pin_memory requested but last batch was unpinned"]
                if config.pin_memory and observed is False
                else [],
            )
            emit("epoch_result", epoch=asdict(epoch), pinning=pinning)
            if error is not None:
                raise error
    finally:
        try:
            close_loader(loader, iterator)
        finally:
            relay_stop.set()
            if relay is not None:
                relay.join(timeout=2)
                if relay.is_alive():
                    raise RuntimeError("worker registration relay did not stop")
            if registrations is not None:
                registrations.close()
                registrations.join_thread()
