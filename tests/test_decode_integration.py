from pathlib import Path

import pytest

from imgread_benchmark.benchmark import BenchmarkImgRead
from imgread_benchmark.cli import main, _normalize_argv
from imgread_benchmark.decode_benchmark import DecodeRunError
from imgread_benchmark import encoded_cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(encoded_cache, "_require_tmpfs", lambda path: None)
    monkeypatch.setattr(encoded_cache, "_memory_budget", lambda: 2**40)
    return tmp_path / "cache"


def test_sequential_counts_warmup_selection_repeats_and_reuse(tmp_path, cache_dir):
    paths = [tmp_path / f"{i}.bin" for i in range(3)]
    for i, path in enumerate(paths):
        path.write_bytes(bytes([i]))
    calls = []

    def decode(data):
        calls.append(bytes(data))
        return data[0]

    benchmark = BenchmarkImgRead(
        filenames=paths,
        func_dict={"custom": decode},
        decode_only=True,
        cache_dir=cache_dir,
        num_repeats=2,
    )
    assert not cache_dir.exists()
    benchmark.run(num_samples=2)
    assert calls == [b"\0"] + [b"\0", b"\1"] * 3
    assert len(benchmark._results["custom"]) == 2
    assert benchmark.decode_report["selected_n"] == 2
    assert not benchmark.decode_report["cache_hit"]
    benchmark.warmup = False
    calls.clear()
    benchmark(num_repeats=1, num_samples=2)
    assert calls == [b"\0", b"\0", b"\1"]
    assert benchmark.decode_report["cache_hit"]


def test_failed_backend_has_no_speed_and_other_backend_finishes(tmp_path, cache_dir):
    source = tmp_path / "sample.bin"
    source.write_bytes(b"x")
    benchmark = BenchmarkImgRead(
        filenames=[source],
        decode_only=True,
        cache_dir=cache_dir,
        func_dict={"bad": lambda _: None, "good": lambda _: 1},
        num_repeats=1,
    )
    with pytest.raises(DecodeRunError, match="returned None"):
        benchmark.run()
    assert set(benchmark.results) == {"good"}
    assert benchmark.decode_report["errors"]["bad"]["stage"] == "initialization"


def test_validation_before_cache_creation(tmp_path, cache_dir):
    benchmark = BenchmarkImgRead(
        filenames=[tmp_path / "missing"],
        decode_only=True,
        cache_dir=cache_dir,
        func_dict={"local": lambda _: 1},
    )
    with pytest.raises(ValueError, match="unavailable"):
        benchmark.run("unknown")
    with pytest.raises(ValueError, match="pickleable"):
        benchmark.run(multiprocessing=True, num_workers=2)
    with pytest.raises(ValueError, match="requires selected"):
        benchmark.run(exclude="local")
    assert not cache_dir.exists()


def test_target_switch_keeps_buffer_registry(cache_dir):
    benchmark = BenchmarkImgRead(filenames=[], decode_only=True, cache_dir=cache_dir)
    benchmark.target_format = "pil"
    from imgread_benchmark.img_libs.PIL import decode_img_pil

    assert benchmark.func_dict["PIL"] is decode_img_pil
    assert not cache_dir.exists()


@pytest.mark.parametrize("prefix", [[], ["benchmark"], ["--decode-only"]])
def test_cli_decode_forms(prefix, cache_dir, capsys):
    source = Path(__file__).parent / "test_imgs"
    args = [
        *prefix,
        str(source),
        "-l",
        "PIL",
        "-n",
        "2",
        "-r",
        "1",
        "--cache-dir",
        str(cache_dir),
    ]
    if "--decode-only" not in args:
        args.append("--decode-only")
    main(args)
    output = capsys.readouterr().out
    assert "decode-only sequential" in output
    assert "Items/sec" in output


@pytest.mark.parametrize(
    "option,value", [("--cache-dir", "/tmp/cache"), ("--cache-limit", "2GiB")]
)
def test_cli_cache_options_require_decode(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        main([".", option, value])
    assert error.value.code == 2
    assert "require --decode-only" in capsys.readouterr().err


def test_cli_corrupt_input_fails(cache_dir, tmp_path, capsys):
    (tmp_path / "bad.jpg").write_bytes(b"bad")
    with pytest.raises(SystemExit) as error:
        main(
            [str(tmp_path), "--decode-only", "--cache-dir", str(cache_dir), "-l", "PIL"]
        )
    assert error.value.code == 1
    assert "decode-only failed" in capsys.readouterr().err


def test_leading_cache_flags():
    args = ["--cache-limit", "2GiB", "--decode-only", "images"]
    assert _normalize_argv(args) == ["benchmark", *args]
