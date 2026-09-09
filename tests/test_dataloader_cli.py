import importlib.util
import json
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
    assert "resource_samples" not in json.loads((output / "result.json").read_text())
    with pytest.raises(FileExistsError):
        write_result(result, output)
    table = render_result(result)
    assert table.columns[2]._cells == ["2"]
    assert table.columns[6]._cells == ["200"]


def test_incomplete_reading_report_and_export(tmp_path):
    from io import StringIO
    from rich.console import Console

    manifest = snapshot_files(make_images(tmp_path / "imgs"))
    config = DataLoaderConfig(epochs=2, batch_size=4)
    result = BenchmarkResult(
        "id",
        "uuid",
        manifest,
        {"requested": config.effective(), "effective": config.effective()},
        epochs=[
            EpochResult.measured(0, 0, 2_000_000_000, 10, 3, 0, "a", "a"),
            EpochResult.measured(
                1, 2_000_000_000, 3_000_000_000, 4, 1, 0, "b", "c", status="failed"
            ),
        ],
        error={
            "reader": "pil-rgb",
            "path": "/images/[red].png",
            "reason": "bad payload",
            "secondary_errors": [{"reason": "shutdown timeout"}],
        },
        warnings=["decoder diagnostic"],
    )
    summary = result.delivery_summary
    assert summary["status"] == "incomplete"
    assert summary["confirmed_deliveries"] == 14
    assert summary["expected_deliveries"] == 20
    assert summary["unconfirmed_deliveries"] == 6
    assert summary["selected_entries_per_epoch"] == 10
    assert summary["successful_epochs"] == 1
    stream = StringIO()
    Console(file=stream, width=240, color_system=None).print(render_result(result))
    text = " ".join(stream.getvalue().split())
    for expected in (
        "INCOMPLETE READING",
        "reader=pil-rgb",
        "14/20",
        "N/A",
        "/images/[red].png",
        "bad payload",
        "shutdown timeout",
        "Warning: decoder diagnostic",
    ):
        assert expected in text
    output = tmp_path / "report"
    write_result(result, output)
    assert read_result(output).to_dict() == result.to_dict()
    assert (
        json.loads((output / "result.json").read_text())["delivery_summary"] == summary
    )


@pytest.mark.parametrize("teardown_failed", [False, True])
def test_completed_reads_and_drop_last_are_not_incomplete(tmp_path, teardown_failed):
    config = DataLoaderConfig(epochs=2, batch_size=4, drop_last=True)
    result = BenchmarkResult(
        "id",
        "uuid",
        snapshot_files(make_images(tmp_path)),
        {"requested": config.effective(), "effective": config.effective()},
        status="failed" if teardown_failed else "success",
        error={"reason": "shutdown timeout"} if teardown_failed else None,
        epochs=[
            EpochResult.measured(i, 0, 1_000_000_000, 8, 2, 2, "a", "b")
            for i in range(2)
        ],
    )
    summary = result.delivery_summary
    assert summary["status"] == "complete"
    assert summary["intentional_drop_last_per_epoch"] == 2
    assert summary["confirmed_deliveries"] == summary["expected_deliveries"] == 16
    assert summary["unconfirmed_deliveries"] == 0
    assert "INCOMPLETE" not in render_result(result).caption.plain


def test_unreported_epoch_does_not_claim_zero_reads(tmp_path):
    config = DataLoaderConfig(epochs=1, batch_size=4)
    result = BenchmarkResult(
        "id",
        "uuid",
        snapshot_files(make_images(tmp_path)),
        {"requested": config.effective(), "effective": config.effective()},
        error={"reason": "consumer crashed before reporting the epoch"},
    )
    summary = result.delivery_summary
    assert summary["reported_epochs"] == 0
    assert summary["confirmed_deliveries"] == 0
    assert summary["unconfirmed_deliveries"] == 10
    assert "not an exact count of unread files" in summary["warning"]


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


@pytest.mark.parametrize("storage", ["files", "memory"])
def test_listing_does_not_run(monkeypatch, capsys, storage):
    from imgread_benchmark.dataloader.models import READER_IDS, ReaderInfo
    from imgread_benchmark.dataloader import readers, runner

    def list_readers(requested_storage):
        assert requested_storage == storage
        return tuple(ReaderInfo(i, i, False, "test unavailable") for i in READER_IDS)

    monkeypatch.setattr(readers, "list_readers", list_readers)
    monkeypatch.setattr(
        runner, "run_benchmark", lambda *args: pytest.fail("listing started a run")
    )
    _build_cli()(["dataloader", "--list-readers", "--storage", storage])
    output = capsys.readouterr().out
    assert all(i in output for i in READER_IDS)
    assert "test unavailable" in output
