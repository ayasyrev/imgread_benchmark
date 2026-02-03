from imgread_benchmark.cli import _normalize_argv


def test_normalize_injects_benchmark_for_path():
    assert _normalize_argv(["/some/path"]) == ["benchmark", "/some/path"]


def test_normalize_does_not_inject_for_known_subcommand():
    assert _normalize_argv(["libs"]) == ["libs"]


def test_normalize_does_not_inject_for_global_flag():
    assert _normalize_argv(["--version"]) == ["--version"]


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
