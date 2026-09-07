import importlib.util
import json
import os
import subprocess
import sys

import pytest

from imgread_benchmark.cli import _build_cli
from imgread_benchmark.dataloader import DataLoaderConfig, snapshot_files
from imgread_benchmark.dataloader.models import BenchmarkResult, EpochResult
from imgread_benchmark.dataloader.report import read_result, render_result, write_result
from tests.dataloader_helpers import make_images


def test_export_roundtrip(tmp_path):
    manifest = snapshot_files(make_images(tmp_path / "imgs", 1))
    cfg = DataLoaderConfig()
    result = BenchmarkResult(
        "id",
        "uuid",
        manifest,
        {"requested": cfg.effective(), "effective": cfg.effective()},
        status="success",
        epochs=[EpochResult.measured(0, 0, 2_000_000_000, 10, 3, 0, "a", "b")],
    )
    result.resource_samples = [{"cpu": None}]
    output = tmp_path / "run"
    write_result(result, output)
    assert read_result(output).to_dict() == result.to_dict()
    assert os.stat(output).st_mode & 0o777 == 0o700
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in output.iterdir())
    assert "resource_samples" not in json.loads((output / "result.json").read_text())
    with pytest.raises(FileExistsError):
        write_result(result, output)
    table = render_result(result)
    assert table.columns[2]._cells == ["2"]
    assert table.columns[6]._cells == ["200"]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["imgs", "--manifest", "m.json"],
        ["imgs", "--num-workers", "-1"],
        ["imgs", "--persistent-workers"],
        ["imgs", "--prefetch-factor", "2"],
    ],
)
def test_invalid_cli(args):
    with pytest.raises(SystemExit) as caught:
        _build_cli()(["dataloader", *args])
    assert caught.value.code == 2


def test_help(capsys):
    with pytest.raises(SystemExit) as caught:
        _build_cli()(["dataloader", "--help"])
    assert caught.value.code == 0
    assert "--num-workers" in capsys.readouterr().out


def test_cli_run_and_manifest(tmp_path):
    pytest.importorskip("torch")
    paths = make_images(tmp_path / "imgs", 3)
    first, second = tmp_path / "a", tmp_path / "b"
    _build_cli()(
        [
            "dataloader",
            str(paths[0].parent),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--output",
            str(first),
        ]
    )
    _build_cli()(
        [
            "dataloader",
            "--manifest",
            str(first / "manifest.json"),
            "--epochs",
            "1",
            "--output",
            str(second),
        ]
    )
    assert read_result(first).manifest == read_result(second).manifest
    assert read_result(first).epochs[0].images_delivered == 3


def test_base_install(tmp_path):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("base_install requires the separate base-only environment")
    assert importlib.util.find_spec("torchvision") is None
    assert importlib.util.find_spec("psutil") is None
    script = "import sys; import imgread_benchmark; from imgread_benchmark.cli import main; main(['--version']); main(['libs']); assert not any(x in sys.modules for x in ('torch','torchvision','psutil'))"
    subprocess.run([sys.executable, "-c", script], check=True)
    paths = make_images(tmp_path, 1)
    result = subprocess.run(
        [sys.executable, "-m", "imgread_benchmark", "dataloader", str(paths[0].parent)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "dataloader extra" in result.stderr


def test_listing_does_not_run(monkeypatch, capsys):
    from imgread_benchmark.dataloader.models import READER_IDS, ReaderInfo
    from imgread_benchmark.dataloader import readers, runner

    monkeypatch.setattr(
        readers,
        "list_readers",
        lambda: tuple(ReaderInfo(i, i, False, "test unavailable") for i in READER_IDS),
    )
    monkeypatch.setattr(
        runner, "run_benchmark", lambda *args: pytest.fail("listing started a run")
    )
    _build_cli()(["dataloader", "--list-readers"])
    output = capsys.readouterr().out
    assert all(i in output for i in READER_IDS)
    assert "test unavailable" in output


def test_acceptance_evaluator_checks_rates_and_identity(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from dataclasses import asdict
    from tests import dataloader_smoke as smoke
    from imgread_benchmark.dataloader.models import digest
    from imgread_benchmark.dataloader.resources import reduce_observations

    monkeypatch.setattr(smoke, "headroom", lambda root: {})
    binding = {"code_sha": "test-sha", "input_hashes": {}}
    smoke.prepare(tmp_path, binding)
    qualified = tmp_path / "test-sha"
    manifest = smoke.load_manifest(qualified / "manifest.json")
    attempt = qualified / "attempt-test"
    smoke.private_dir(attempt)
    smoke.save(attempt / "outcome.json", {"status": "success"})
    for index, raw in enumerate(smoke.matrix()):
        config = DataLoaderConfig(**raw)
        config_id = digest(
            dict(config=config.effective(), selection_id=manifest.selection_id)
        )
        result = BenchmarkResult(
            config_id,
            f"execution-{index}",
            manifest,
            {"requested": asdict(config), "effective": config.effective()},
            status="success",
        )
        result.consumer = dict(
            pid=index + 100,
            started_wall_time=1,
            os_create_time=1 if config.monitor_resources else None,
        )
        result.pinning = {"status": "not_requested"}
        for e in range(3):
            order_id = digest(
                torch.randperm(10, generator=torch.Generator().manual_seed(e)).tolist()
            )
            result.epochs.append(
                EpochResult.measured(
                    e,
                    (1 + 2 * e) * 10**9,
                    (2 + 2 * e) * 10**9,
                    10,
                    3,
                    0,
                    order_id,
                    order_id,
                )
            )
        if config.monitor_resources:
            result.baseline = {"method": "test", "sweep_end_ns": 0}
            result.preparation["worker_registrations"] = []
            _, _, summaries = reduce_observations(
                [], result.epochs, (), config.num_workers
            )
            for epoch, summary in zip(result.epochs, summaries):
                epoch.resources = summary
            result.resource_status = "partial"
        logical_id = digest(
            dict(
                code_sha=binding["code_sha"],
                selection_id=manifest.selection_id,
                effective_config=config.effective(),
                matrix_position=index,
            )
        )
        write_result(result, attempt / f"{index:02}-{logical_id}")
    smoke.verify(tmp_path, binding)
    assert json.loads((attempt / "verification.json").read_text())["executions"] == 20
    path = sorted(attempt.glob("*/result.json"))[0]
    data = json.loads(path.read_text())
    data["epochs"][0]["images_per_second"] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        smoke.verify(tmp_path, binding)
