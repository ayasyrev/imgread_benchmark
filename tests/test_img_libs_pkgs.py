import builtins
import importlib
import importlib.util
import sys
import types

import imgread_benchmark.img_libs.img_libs_pkgs as img_libs_pkgs


def test_img_lib_available_skips_broken_jpeg4py(monkeypatch):
    def fake_find_spec(name):
        if name in ("jpeg4py", "PIL"):
            return object()
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jpeg4py":
            raise OSError("libjpeg missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    reloaded = importlib.reload(img_libs_pkgs)

    assert "jpeg4py" not in reloaded.img_lib_available
    assert "PIL" in reloaded.img_lib_available


def test_is_jpeg4py_usable_true(monkeypatch):
    decode_count = [0]
    path_holder = [""]

    class FakeJPEG:
        def __init__(self, path):
            path_holder[0] = path

        def decode(self):
            decode_count[0] += 1
            return b"ok"

    module = types.ModuleType("jpeg4py")
    setattr(module, "JPEG", FakeJPEG)
    monkeypatch.setitem(sys.modules, "jpeg4py", module)

    assert img_libs_pkgs._is_jpeg4py_usable() is True
    assert decode_count[0] == 1
    assert isinstance(path_holder[0], str)


def test_is_jpeg4py_usable_false_on_decode_error(monkeypatch):
    class FakeJPEG:
        def __init__(self, path):
            pass

        def decode(self):
            raise OSError("libjpeg missing")

    module = types.ModuleType("jpeg4py")
    setattr(module, "JPEG", FakeJPEG)
    monkeypatch.setitem(sys.modules, "jpeg4py", module)

    assert img_libs_pkgs._is_jpeg4py_usable() is False
