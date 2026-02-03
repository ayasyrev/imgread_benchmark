import builtins
import importlib
import importlib.util
import sys
import types


def load_img_libs_pkgs(monkeypatch, find_library_result="libjpeg", jpeg4py_module=None):
    import ctypes.util

    real_find_spec = importlib.util.find_spec

    monkeypatch.setattr(ctypes.util, "find_library", lambda name: find_library_result)
    sys.modules.pop("jpeg4py", None)
    sys.modules.pop("jpeg4py._py", None)
    sys.modules.pop("imgread_benchmark.img_libs.img_libs_pkgs", None)
    if jpeg4py_module is not None:

        def fake_find_spec(name):
            if name == "jpeg4py":
                return object()
            return real_find_spec(name)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setitem(sys.modules, "jpeg4py", jpeg4py_module)
    import imgread_benchmark.img_libs.img_libs_pkgs as img_libs_pkgs

    return img_libs_pkgs


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

    img_libs_pkgs = load_img_libs_pkgs(monkeypatch)
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
    img_libs_pkgs = load_img_libs_pkgs(monkeypatch, jpeg4py_module=module)

    assert img_libs_pkgs._is_jpeg4py_usable() is True
    decode_count[0] = 0
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
    img_libs_pkgs = load_img_libs_pkgs(monkeypatch, jpeg4py_module=module)

    assert img_libs_pkgs._is_jpeg4py_usable() is False


def test_is_jpeg4py_usable_false_without_libjpeg(monkeypatch):
    decode_count = [0]

    class FakeJPEG:
        def __init__(self, path):
            pass

        def decode(self):
            decode_count[0] += 1
            return b"ok"

    module = types.ModuleType("jpeg4py")
    setattr(module, "JPEG", FakeJPEG)
    img_libs_pkgs = load_img_libs_pkgs(
        monkeypatch, find_library_result=None, jpeg4py_module=module
    )
    assert img_libs_pkgs._is_jpeg4py_usable() is False
    assert decode_count[0] == 0


def test_get_img_lib_available():
    """Test get_img_lib_available function."""
    from imgread_benchmark.img_libs.img_libs_pkgs import get_img_lib_available

    libs = get_img_lib_available()
    assert isinstance(libs, list)
    # PIL should always be available (it's a required dependency)
    assert "PIL" in libs


def test_get_img_lib_available_cached():
    """Test that get_img_lib_available is cached."""
    from imgread_benchmark.img_libs.img_libs_pkgs import get_img_lib_available

    libs1 = get_img_lib_available()
    libs2 = get_img_lib_available()
    assert libs1 is libs2


def test_lazy_list_is_sequence():
    import collections.abc

    from imgread_benchmark.img_libs.img_libs_pkgs import img_lib_available

    assert isinstance(img_lib_available, collections.abc.Sequence)
