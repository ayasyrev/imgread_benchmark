from imgread_benchmark import benchmark as benchmark_mod


def test_get_read_to_format_returns_dicts():
    read_to_format = benchmark_mod._get_read_to_format()
    assert isinstance(read_to_format["def"], dict)
    assert isinstance(read_to_format["pil"], dict)
    assert isinstance(read_to_format["np"], dict)


def test_benchmark_img_read_with_dummy_funcs():
    def _dummy_reader(_path: str) -> str:
        return "ok"

    bench = benchmark_mod.BenchmarkImgRead(
        filenames=["/tmp/not_used.jpg"],
        func_dict={"dummy": _dummy_reader},
        target_format="def",
        num_repeats=1,
        clear_progress=True,
    )

    assert bench.func_dict["dummy"]("/tmp/x.jpg") == "ok"
    assert "dummy" in bench.func_names
