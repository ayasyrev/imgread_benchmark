import builtins
import importlib

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
