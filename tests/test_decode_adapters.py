from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from imgread_benchmark import read_img


@pytest.mark.parametrize("target", ["def", "pil", "np"])
def test_file_buffer_equivalence(target):
    functions, unsupported = read_img.get_decode_functions(target)
    assert "PIL" in functions
    assert not set(functions) & set(unsupported)
    suffix = {"def": "", "pil": "_pil", "np": "_ndarray"}[target]
    filename = Path(__file__).parent / "test_imgs/cat.jpg"
    data = memoryview(filename.read_bytes())
    assert data.readonly
    for name, decoder in functions.items():
        adapter = read_img.get_img_libs()[name]
        expected = getattr(adapter, "read_img" + suffix)(str(filename))
        actual = decoder(data)
        assert type(actual) is type(expected), name
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected), err_msg=name
        )
        with pytest.raises(Exception):
            decoder(memoryview(b"this is not an image"))


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_png_preserves_each_adapters_semantics(tmp_path, mode):
    filename = tmp_path / "sample.png"
    Image.new(mode, (13, 17)).save(filename)
    functions, _ = read_img.get_decode_functions("np")
    for name in ("PIL", "cv2", "imageio", "skimage", "imagecodecs", "imgread"):
        if name not in functions:
            continue
        expected = read_img.get_img_libs()[name].read_img_ndarray(str(filename))
        actual = functions[name](memoryview(filename.read_bytes()))
        np.testing.assert_array_equal(actual, expected, err_msg=name)


def test_registry_requires_explicit_buffer_api_and_does_not_hide_errors(monkeypatch):
    def fail(data):
        raise OSError("native decode failure")

    monkeypatch.setattr(
        read_img,
        "get_img_libs",
        lambda: {
            "file_only": SimpleNamespace(read_img=lambda path: None),
            "plugin": SimpleNamespace(decode_img=fail),
            "invalid": SimpleNamespace(decode_img=17),
        },
    )
    functions, reasons = read_img.get_decode_functions()
    assert set(functions) == {"plugin"}
    assert set(reasons) == {"file_only", "invalid"}
    with pytest.raises(OSError, match="native decode failure"):
        functions["plugin"](b"data")


def test_lazy_pil_output_is_materialized_before_timer_stops():
    from imgread_benchmark.decode_benchmark import _sequential_pass

    events = []

    class LazyImage(Image.Image):
        def load(self):
            events.append("decode")

    def clock():
        events.append("clock")
        return len(events)

    assert _sequential_pass(lambda data: LazyImage(), [b"x"], [0], clock) == 2
    assert events == ["clock", "decode", "clock"]


@pytest.mark.parametrize(
    "adapter, package",
    [
        ("ajpegli", "ajpegli"),
        ("imgread", "imgread"),
        ("imgread_rs", "imgread_rs"),
        ("local_rs", "local_rs"),
    ],
)
def test_missing_native_bytes_api_is_unavailable(monkeypatch, adapter, package):
    import importlib.util
    import sys

    monkeypatch.setitem(sys.modules, package, SimpleNamespace())
    filename = Path(read_img.__file__).parent / "img_libs" / f"{adapter}.py"
    spec = importlib.util.spec_from_file_location(f"test_missing_{adapter}", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(read_img, "get_img_libs", lambda: {adapter: module})
    functions, reasons = read_img.get_decode_functions()
    assert not functions and adapter in reasons
