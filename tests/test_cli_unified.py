from imgread_benchmark.cli import _normalize_argv


def test_normalize_injects_benchmark_for_path():
    assert _normalize_argv(["/some/path"]) == ["benchmark", "/some/path"]


def test_normalize_does_not_inject_for_known_subcommand():
    assert _normalize_argv(["libs"]) == ["libs"]


def test_normalize_does_not_inject_for_global_flag():
    assert _normalize_argv(["--version"]) == ["--version"]


def test_normalize_argv_injects_benchmark_with_leading_flags():
    argv = ["-n", "10", "-t", "np", "/imgs"]
    assert _normalize_argv(argv) == ["benchmark", "-n", "10", "-t", "np", "/imgs"]


def test_normalize_argv_injects_benchmark_with_flag_without_value():
    argv = ["-A", "/imgs"]
    assert _normalize_argv(argv) == ["benchmark", "-A", "/imgs"]


def test_normalize_argv_keeps_root_help():
    assert _normalize_argv(["--help"]) == ["--help"]


def test_normalize_argv_keeps_root_version():
    assert _normalize_argv(["-V"]) == ["-V"]


def test_normalize_argv_does_not_inject_for_unknown_flag():
    assert _normalize_argv(["--unknown", "/imgs"]) == ["--unknown", "/imgs"]


def test_cli_version_prints_package_version(capsys):
    from imgread_benchmark.cli import main

    main(["--version"])
    out = capsys.readouterr().out.strip()
    assert out


def test_cli_data_does_not_import_jpeg4py(monkeypatch):
    import builtins

    seen = {"jpeg4py": False}
    orig = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "jpeg4py" or name.startswith("jpeg4py."):
            seen["jpeg4py"] = True
            raise ImportError("blocked")
        return orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)

    from unittest.mock import patch, MagicMock

    from imgread_benchmark.cli import main

    mock_provider_cls = MagicMock()
    mock_provider = mock_provider_cls.return_value

    with patch.dict(
        "imgread_benchmark.cl_data.DATASET_PROVIDERS", {"imagenette": mock_provider_cls}
    ):
        main(["data", "imagenette", "--size", "160"])
        mock_provider.download.assert_called_once_with(size="160")

    assert not seen["jpeg4py"]


def test_cli_libs_prints_versions(monkeypatch, capsys):
    from imgread_benchmark.cli import main

    monkeypatch.setattr(
        "imgread_benchmark.img_libs.img_libs_pkgs.get_img_lib_available",
        lambda: ["PIL", "cv2"],
    )
    monkeypatch.setattr(
        "imgread_benchmark.read_img.get_read_img_version",
        lambda: {"PIL": "10.0.0", "cv2": "4.10.0"},
    )

    main(["libs"])
    out = capsys.readouterr().out
    assert "PIL" in out and "10.0.0" in out
    assert "cv2" in out and "4.10.0" in out


import pytest
from imgread_benchmark.cli import _build_cli


def test_benchmark_missing_path_errors_to_stderr(tmp_path, capsys):
    cli = _build_cli()
    with pytest.raises(SystemExit) as exc:
        cli(["benchmark", str(tmp_path / "missing")])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_data_unknown_dataset_errors_to_stderr(capsys):
    cli = _build_cli()
    with pytest.raises(SystemExit) as exc:
        cli(["data", "nope"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Unknown dataset" in captured.err
