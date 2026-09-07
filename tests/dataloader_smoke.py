"""Fixed post-review acceptance contract. This is not a benchmark scheduler.

Run from the canonical worktree with locked extras. All records remain external.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from imgread_benchmark.dataloader import (
    DataLoaderConfig,
    load_manifest,
    read_result,
    run_benchmark,
    snapshot_files,
    write_result,
)
from imgread_benchmark.dataloader.models import READER_IDS, canonical_json, digest
from imgread_benchmark.dataloader.report import write_private

ID = "20260907_143308_dataloader-benchmark"
CANONICAL = Path("/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark")
RECORDS = Path("/home/aya/.codex/workflow-records/imgread-benchmark") / ID
INPUTS = Path("/home/aya/Prj/imgread_benchmark/docs/plans")
DEFAULT_ROOT = Path("/tmp/imgread-dataloader-acceptance") / ID


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def contained(path, root):
    return path.resolve().is_relative_to(root.resolve())


def field(text, key):
    match = re.search(
        r"^\|\s*" + re.escape(key) + r"\s*\|\s*`?([^|`]+)`?\s*\|", text, re.MULTILINE
    )
    return match.group(1).strip() if match else None


def guard(root):
    import imgread_benchmark

    cwd = Path.cwd().resolve()
    if (
        cwd != CANONICAL.resolve()
        or Path(git("rev-parse", "--show-toplevel")).resolve() != cwd
    ):
        raise RuntimeError("run from the canonical worktree")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("acceptance requires a completely clean committed worktree")
    if not contained(Path(imgread_benchmark.__file__), cwd):
        raise RuntimeError("import origin escaped reviewed worktree")
    for path in [
        Path("src"),
        Path("tests"),
        *Path("src/imgread_benchmark").rglob("*.py"),
        *Path("tests").glob("*.py"),
    ]:
        if not contained(path, cwd):
            raise RuntimeError(f"runnable path escaped: {path}")
    for path in (root, RECORDS, RECORDS / "implementation-reviews"):
        if contained(path, cwd) or contained(path, INPUTS.parent.parent):
            raise RuntimeError(f"external path aliases a checkout: {path}")
    if contained(RECORDS, root) or contained(root, RECORDS):
        raise RuntimeError("workflow records and experiment artifacts must be separate")
    code_sha = git("rev-parse", "HEAD")
    hashes = {}
    for name in ("task.md", "task.review-01.md", "plan.md", "plan.review-01.md"):
        source = INPUTS / f"{ID}.{name}"
        copy = Path("docs/plans") / source.name
        git("ls-files", "--error-unmatch", str(copy))
        if source.read_bytes() != copy.read_bytes():
            raise RuntimeError(f"workflow input copy differs: {source}")
        hashes[name] = sha(source)
    for kind in ("task", "plan"):
        review = (INPUTS / f"{ID}.{kind}.review-01.md").read_text()
        if (
            field(review, "Verdict") != "approved"
            or field(review, "Bound artifact SHA-256") != hashes[kind + ".md"]
        ):
            raise RuntimeError(f"unbound {kind} approval")
    reviews = sorted(
        (RECORDS / "implementation-reviews").glob(f"{ID}.implementation.review-*.md")
    )
    approval = None
    for review in reviews:
        content = review.read_text()
        if (
            field(content, "Verdict") == "approved"
            and field(content, "Reviewed code SHA") == code_sha
        ):
            if (
                hashes["task.md"] in content
                and hashes["plan.md"] in content
                and str(cwd) in content
            ):
                approval = review
    if approval is None:
        raise RuntimeError("no bound approved implementation review for this exact SHA")
    return dict(
        code_sha=code_sha,
        input_hashes=hashes,
        approval=str(approval),
        approval_sha256=sha(approval),
        import_origin=imgread_benchmark.__file__,
    )


def headroom(root):
    import psutil

    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or sys.version_info[:2] != (3, 13)
    ):
        raise RuntimeError("acceptance requires Linux x86_64 / Python 3.13")
    from importlib.metadata import version

    for package, expected in (
        ("torch", "2.10.0"),
        ("torchvision", "0.25.0"),
        ("opencv-python-headless", "4.13.0.90"),
    ):
        if version(package).split("+")[0] != expected:
            raise RuntimeError(f"dependency mismatch: {package}")
    values = dict(
        logical_cpus=psutil.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        available_ram=psutil.virtual_memory().available,
        artifact_free=shutil.disk_usage(root if root.exists() else "/tmp").free,
        shm_free=shutil.disk_usage("/dev/shm").free,
    )
    values["available_cpus"] = len(values["cpu_affinity"])
    minimums = dict(
        available_cpus=4,
        available_ram=8 * 2**30,
        artifact_free=2 * 2**30,
        shm_free=256 * 2**20,
    )
    for key, minimum in minimums.items():
        if values[key] < minimum:
            raise RuntimeError(
                f"resource envelope unmet: {key}={values[key]} < {minimum}"
            )
    return values


def matrix():
    rows = []
    for reader in READER_IDS:
        for workers, persistent in ((0, False), (2, True)):
            for monitor in (False, True):
                rows.append(
                    DataLoaderConfig(
                        reader=reader,
                        num_workers=workers,
                        persistent_workers=persistent,
                        monitor_resources=monitor,
                        epochs=3,
                        batch_size=4,
                        shuffle=True,
                    )
                )
    a = DataLoaderConfig(epochs=3, batch_size=4, shuffle=True, monitor_resources=True)
    b = DataLoaderConfig(
        reader="cv2-bgr-cvtcolor",
        num_workers=2,
        persistent_workers=True,
        epochs=3,
        batch_size=4,
        shuffle=True,
        monitor_resources=True,
    )
    return [asdict(row) for row in [*rows, a, b, b, a]]


def private_dir(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=False)


def save(path, value):
    write_private(path, canonical_json(value) + "\n")


def prepare(root, binding):
    import numpy as np
    from PIL import Image

    qualified = root / binding["code_sha"]
    if qualified.exists():
        verify_data(qualified)
        if json.loads((qualified / "matrix.json").read_text()) != matrix():
            raise RuntimeError("existing matrix differs")
        return
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    private_dir(qualified)
    private_dir(qualified / "data")
    y, x = np.indices((320, 480))
    paths = []
    for i in range(10):
        array = np.stack(
            ((x + i) % 256, (y + 3 * i) % 256, (x + y + 7 * i) % 256), -1
        ).astype("uint8")
        path = qualified / "data" / f"{i:03}.png"
        Image.fromarray(array).save(path, format="PNG")
        path.chmod(0o600)
        paths.append(path)
    manifest = snapshot_files(paths)
    save(qualified / "manifest.json", manifest.to_dict())
    save(qualified / "matrix.json", matrix())
    save(
        qualified / "environment.json",
        dict(
            **binding,
            headroom=headroom(root),
            generator_sha=sha(__file__),
            lock_sha=sha("uv.lock"),
            data_hashes={path.name: sha(path) for path in paths},
            manifest_sha=sha(qualified / "manifest.json"),
        ),
    )


def verify_data(qualified):
    environment = json.loads((qualified / "environment.json").read_text())
    if (
        environment["generator_sha"] != sha(__file__)
        or environment["lock_sha"] != sha("uv.lock")
        or environment["manifest_sha"] != sha(qualified / "manifest.json")
    ):
        raise RuntimeError("generator/lock/manifest content changed")
    expected = {f"{i:03}.png" for i in range(10)}
    if (
        set(environment["data_hashes"]) != expected
        or {p.name for p in (qualified / "data").iterdir()} != expected
    ):
        raise RuntimeError("data file set changed")
    for name, expected_sha in environment["data_hashes"].items():
        if sha(qualified / "data" / name) != expected_sha:
            raise RuntimeError(f"data changed: {name}")


def ensure_no_consumers():
    """A lost supervisor releases flock; explicitly reject its surviving processes."""
    import psutil

    survivors = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd", "status", "uids"]):
        info = process.info
        if (
            not info["uids"]
            or info["uids"].real != os.getuid()
            or info["status"] == psutil.STATUS_ZOMBIE
        ):
            continue
        args = info["cmdline"] or []
        related = "imgread_benchmark.dataloader._child" in args or any(
            "from multiprocessing.spawn import spawn_main" in arg
            or "from multiprocessing.resource_tracker import main" in arg
            for arg in args
        )
        if related and (
            info["cwd"] is None or Path(info["cwd"]).resolve() == CANONICAL.resolve()
        ):
            survivors.append(info["pid"])
    if survivors:
        raise RuntimeError(
            f"previous/active consumer group has surviving PIDs {survivors}; stop and confirm cleanup before a new attempt"
        )


def save_execution(output, logical_id, position, raw, result=None, error=None):
    if result is not None:
        write_result(result, output)
    else:
        private_dir(output)
        save(
            output / "failure.json",
            dict(
                stage="acceptance",
                reader=raw["reader"],
                path=None,
                reason=f"{type(error).__name__}: {error}",
            ),
        )
    save(
        output / "invocation.json",
        dict(
            logical_id=logical_id,
            matrix_position=position,
            config=raw,
            api="run_benchmark(manifest, config, timeout_seconds=60)",
            execution_id=result.execution_id if result else None,
            env={
                k: os.environ.get(k)
                for k in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        ),
    )
    write_private(output / "stdout.txt", "")
    write_private(
        output / "stderr.txt", "\n".join(result.warnings) if result else str(error)
    )


def durable_record(kind, attempt, binding, value):
    # Control records must remain outside both checkouts and the artifact tree.
    if (
        contained(RECORDS, CANONICAL)
        or contained(RECORDS, INPUTS.parent.parent)
        or contained(RECORDS, attempt.parent.parent)
    ):
        raise RuntimeError("workflow record root containment failed")
    save(
        RECORDS / f"{binding['code_sha']}-{attempt.name}-{kind}.json",
        dict(**binding, attempt=str(attempt), **value),
    )


def run(root, binding):
    import fcntl

    qualified = root / binding["code_sha"]
    verify_data(qualified)
    if json.loads((qualified / "matrix.json").read_text()) != matrix():
        raise RuntimeError("matrix changed")
    # One active attempt across invocations. No retry/resume or result reuse.
    with open(
        os.open(root / "active.lock", os.O_CREAT | os.O_RDWR, 0o600), "w"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_no_consumers()
        attempt = qualified / f"attempt-{uuid.uuid4()}"
        private_dir(attempt)
        save(
            attempt / "launch.json",
            dict(
                **binding,
                argv=sys.argv,
                env={
                    k: os.environ.get(k)
                    for k in (
                        "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                },
                headroom=headroom(root),
            ),
        )
        durable_record(
            "launch", attempt, binding, dict(launch_sha=sha(attempt / "launch.json"))
        )
        manifest = load_manifest(qualified / "manifest.json")
        outcome = "failed"
        try:
            start = time.monotonic()
            for position, raw in enumerate(matrix()):
                if time.monotonic() - start >= 900:
                    raise TimeoutError("S2 total budget exceeded")
                cfg = DataLoaderConfig(**raw)
                logical_id = digest(
                    dict(
                        code_sha=binding["code_sha"],
                        selection_id=manifest.selection_id,
                        effective_config=cfg.effective(),
                        matrix_position=position,
                    )
                )
                output = attempt / f"{position:02}-{logical_id}"
                try:
                    result = run_benchmark(manifest, cfg, timeout_seconds=60)
                except BaseException as exc:
                    save_execution(
                        output,
                        logical_id,
                        position,
                        raw,
                        getattr(exc, "result", None),
                        exc,
                    )
                    raise
                save_execution(output, logical_id, position, raw, result)
            outcome = "success"
        finally:
            save(
                attempt / "outcome.json",
                dict(status=outcome, code_sha=binding["code_sha"]),
            )
            durable_record(
                "outcome",
                attempt,
                binding,
                dict(
                    status=outcome,
                    artifact_hashes={
                        str(p.relative_to(attempt)): sha(p)
                        for p in attempt.rglob("*")
                        if p.is_file()
                    },
                ),
            )
        print(attempt)


def verify(root, binding):
    import torch
    from imgread_benchmark.dataloader.resources import reduce_observations

    qualified = root / binding["code_sha"]
    verify_data(qualified)
    attempts = sorted(qualified.glob("attempt-*"), key=lambda p: p.stat().st_mtime_ns)
    if not attempts:
        raise RuntimeError("no attempt")
    attempt = attempts[-1]
    if json.loads((attempt / "outcome.json").read_text())["status"] != "success":
        raise RuntimeError("latest attempt failed/incomplete")
    directories = sorted(p for p in attempt.iterdir() if p.is_dir())
    if len(directories) != 20:
        raise RuntimeError("expected exactly 20 executions")
    results = [read_result(directory) for directory in directories]
    identities = set()
    execution_ids = set()
    workload = load_manifest(qualified / "manifest.json")
    for position, (result, config) in enumerate(zip(results, matrix())):
        assert result.status == "success" and result.config["requested"] == config
        assert result.manifest == workload and len(result.epochs) == 3
        assert result.config_id == digest(
            dict(
                config=DataLoaderConfig(**config).effective(),
                selection_id=workload.selection_id,
            )
        )
        logical_id = digest(
            dict(
                code_sha=binding["code_sha"],
                selection_id=workload.selection_id,
                effective_config=DataLoaderConfig(**config).effective(),
                matrix_position=position,
            )
        )
        assert directories[position].name == f"{position:02}-{logical_id}"
        identity = (
            result.consumer["pid"],
            result.consumer["os_create_time"] or result.consumer["started_wall_time"],
        )
        assert identity not in identities
        identities.add(identity)
        assert result.execution_id not in execution_ids
        execution_ids.add(result.execution_id)
        assert result.pinning["status"] == "not_requested"
        assert [e.index for e in result.epochs] == [0, 1, 2]
        for e in result.epochs:
            assert (
                e.status,
                e.images_delivered,
                e.batches_delivered,
                e.images_dropped,
            ) == ("success", 10, 3, 0)
            assert math.isclose(e.images_per_second, 10 / e.epoch_seconds, rel_tol=1e-9)
            assert math.isclose(
                e.ms_per_image, 1000 * e.epoch_seconds / 10, rel_tol=1e-9
            )
            assert math.isclose(e.batches_per_second, 3 / e.epoch_seconds, rel_tol=1e-9)
            assert (
                e.order_id == results[0].epochs[e.index].order_id
                and e.delivered_order_id == e.order_id
            )
            expected_order = torch.randperm(
                10, generator=torch.Generator().manual_seed(e.index)
            ).tolist()
            assert e.order_id == digest(expected_order)
        assert (
            result.late_epoch_seconds_median
            == sum(e.epoch_seconds for e in result.epochs[1:]) / 2
        )
        if config["monitor_resources"]:
            assert result.baseline and result.baseline["method"]
            assert result.baseline["sweep_end_ns"] <= result.epochs[0].start_ns
            assert result.consumer["os_create_time"] is not None
            _, recomputed_intervals, recomputed_summaries = reduce_observations(
                result.resource_samples,
                result.epochs,
                result.preparation["worker_registrations"],
                config["num_workers"],
            )
            assert result.resource_intervals == recomputed_intervals
            assert [e.resources for e in result.epochs] == recomputed_summaries
            for sample in result.resource_samples:
                if sample["epoch"] is not None:
                    e = result.epochs[sample["epoch"]]
                    assert (
                        e.start_ns
                        <= sample["sweep_start_ns"]
                        <= sample["sweep_end_ns"]
                        <= e.end_ns
                    )
            for interval in result.resource_intervals:
                if interval["included"]:
                    e = result.epochs[interval["epoch"]]
                    assert (
                        e.start_ns
                        <= interval["start_ns"]
                        < interval["end_ns"]
                        <= e.end_ns
                    )
                    assert "cross-boundary" not in interval["reasons"]
        else:
            assert result.resource_status == "off" and result.baseline is None
            assert not result.resource_samples and not result.resource_intervals
    verification = dict(
        status="success",
        executions=20,
        **binding,
        artifact_hashes={
            str(p.relative_to(attempt)): sha(p)
            for p in attempt.rglob("*")
            if p.is_file()
        },
    )
    save(attempt / "verification.json", verification)
    durable_record(
        "verification",
        attempt,
        binding,
        dict(status="success", verification_sha=sha(attempt / "verification.json")),
    )
    print(canonical_json(verification))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if (
        args.root != DEFAULT_ROOT
        or ".." in args.root.parts
        or args.root.resolve() != args.root
    ):
        raise ValueError("acceptance root is fixed and must not use path aliases")
    os.umask(0o077)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    binding = guard(args.root)
    headroom(args.root)
    budget = 900 if args.command == "run" else 60

    def expired(signum, frame):
        raise TimeoutError(f"{args.command} stage budget exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(budget)
    try:
        globals()[args.command](args.root, binding)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
