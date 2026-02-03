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
