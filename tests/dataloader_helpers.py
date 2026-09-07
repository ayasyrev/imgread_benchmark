"""Spawn-safe diagnostic datasets and deterministic image fixtures."""

from pathlib import Path

import numpy as np
from PIL import Image


def make_images(root, count=10, shape=(32, 48)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = root / f"{index:03}.png"
        Image.fromarray(np.full((*shape, 3), index, dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


class IndexDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        assert target == 0
        return int(image[0, 0, 0])


def independent_orders(root, reader):
    import torch
    from torch.utils.data import DataLoader
    from imgread_benchmark.dataloader import (
        EpochSampler,
        ImageListDataset,
        snapshot_files,
    )
    from imgread_benchmark.dataloader.engine import close_loader

    manifest = snapshot_files([Path(root) / f"{i:03}.png" for i in range(10)])
    dataset = IndexDataset(ImageListDataset(manifest, reader, False))
    sampler = EpochSampler(10, 19, True)
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    orders = []
    try:
        for epoch in range(3):
            sampler.set_epoch(epoch)
            orders.append([int(i) for batch in loader for i in batch])
            assert (
                orders[-1]
                == torch.randperm(
                    10, generator=torch.Generator().manual_seed(19 + epoch)
                ).tolist()
            )
    finally:
        close_loader(loader, loader._iterator)
    return orders


class SlowImageDataset:
    """Only test consumers sleep; production iteration has no delay hook."""

    def __init__(self, manifest, reader_id, geometry=True):
        self.manifest, self.reader_id, self.geometry = manifest, reader_id, geometry

    def __len__(self):
        return self.manifest.selected_n

    def __getitem__(self, index):
        import time
        from imgread_benchmark.dataloader.dataset import ImageTransform
        from imgread_benchmark.dataloader.readers import get_reader

        time.sleep(0.15)
        return ImageTransform(self.geometry)(
            get_reader(self.reader_id)(self.manifest.paths[index])
        ), 0


def stalled_teardown_child(mode):
    """Report terminal state, then keep stdout open and ignore graceful shutdown."""
    import json
    import os
    import signal
    import sys
    import time

    from imgread_benchmark.dataloader.models import EpochResult
    from dataclasses import asdict

    request = json.loads(sys.stdin.readline())
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def emit(event, **data):
        print(
            json.dumps(
                dict(
                    event=event,
                    protocol_version=1,
                    config_id=request["config_id"],
                    **data,
                )
            ),
            flush=True,
        )

    if mode == "done":
        emit("done", status="success")
    elif mode == "run_error":
        emit(
            "run_error",
            error={
                "reason": "injected read failure",
                "reader": "pil-rgb",
                "path": request["manifest"]["entries"][0]["path"],
                "stage": "read",
            },
        )
    elif mode == "eof":
        os.close(sys.stdout.fileno())
    else:
        status = "failed" if mode == "failed_epoch" else "success"
        epoch = EpochResult.measured(
            0, 1, 2, 1, 1, 0, "order", "delivered", status=status
        )
        emit("epoch_result", epoch=asdict(epoch), pinning={})
    while True:
        time.sleep(0.05)


def teardown_caller(root, mode):
    """Exercise production supervision with no configuration timeout or caller signal."""
    import os
    import signal
    import time
    from imgread_benchmark.dataloader import (
        BenchmarkRunError,
        DataLoaderConfig,
        run_benchmark,
        snapshot_files,
        readers,
        runner,
    )
    from imgread_benchmark.dataloader.models import ReaderInfo

    original = runner.subprocess.Popen

    def launch(command, *args, **kwargs):
        assert command[1:] == ["-m", "imgread_benchmark.dataloader._child"]
        process = original(
            [command[0], "-m", "tests.dataloader_helpers", "stalled-child", mode],
            *args,
            **kwargs,
        )
        (root / "consumer.pid").write_text(str(process.pid))
        return process

    runner.subprocess.Popen = launch
    readers.probe_reader = lambda reader: ReaderInfo(reader, "test", True)
    runner._SHUTDOWN_GRACE_SECONDS = 0.2
    runner._SHUTDOWN_TERM_SECONDS = 0.2
    original_sigterm = signal.getsignal(signal.SIGTERM)
    start = time.monotonic()
    try:
        run_benchmark(
            snapshot_files([root / "000.png"]),
            DataLoaderConfig(epochs=2 if mode == "failed_epoch" else 1),
        )
    except BenchmarkRunError as exc:
        assert exc.result.status == "failed"
        errors = [exc.result.error, *exc.result.error.get("secondary_errors", [])]
        assert any("shutdown timeout" in error["reason"] for error in errors)
        if mode == "run_error":
            assert exc.result.error["reason"] == "injected read failure"
            assert exc.result.error["path"] == str(root / "000.png")
        assert time.monotonic() - start < 8
        assert not runner.group_members(int((root / "consumer.pid").read_text()))
        assert signal.getsignal(signal.SIGTERM) == original_sigterm
        return dict(error=exc.result.error, caller_pid=os.getpid())
    raise AssertionError("stalled teardown was accepted")


if __name__ == "__main__":
    import json
    import sys

    if sys.argv[1] == "slow-child":
        from imgread_benchmark.dataloader import dataset, _child

        dataset.ImageListDataset = SlowImageDataset
        raise SystemExit(_child.main())
    if sys.argv[1] == "stalled-child":
        stalled_teardown_child(sys.argv[2])
    elif sys.argv[1] == "teardown-caller":
        print(json.dumps(teardown_caller(Path(sys.argv[2]), sys.argv[3])))
        raise SystemExit(0)
    print(json.dumps(independent_orders(sys.argv[1], sys.argv[2])))
