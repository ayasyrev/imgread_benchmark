from importlib.metadata import PackageNotFoundError
import types


def _clear_caches():
    from imgread_benchmark.img_libs import img_libs_pkgs
    from imgread_benchmark import read_img

    img_libs_pkgs.get_plugin_entry_points.cache_clear()
    img_libs_pkgs.get_lib_package_map.cache_clear()
    img_libs_pkgs.get_img_lib_available.cache_clear()
    read_img.get_img_libs.cache_clear()
    read_img.get_read_img.cache_clear()
    read_img.get_read_img_pil.cache_clear()
    read_img.get_read_img_ndarray.cache_clear()
    read_img.get_read_img_version.cache_clear()


def test_plugin_adapter_loaded_and_callable(monkeypatch):
    from imgread_benchmark.img_libs import img_libs_pkgs
    from imgread_benchmark import read_img

    class FakeDist:
        name = "my-cool-backend"

    class FakeEntryPoint:
        name = "my_backend"
        value = "my_backend_plugin:adapter"
        module = "my_backend_plugin"
        dist = FakeDist()

        def load(self):
            return types.SimpleNamespace(
                read_img=lambda _path: "r",
                read_img_pil=lambda _path: "p",
                read_img_ndarray=lambda _path: "n",
            )

    real_find_spec = img_libs_pkgs.find_spec

    def fake_find_spec(name):
        if name == "my_backend_plugin":
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(
        img_libs_pkgs, "entry_points", lambda **kwargs: [FakeEntryPoint()]
    )
    monkeypatch.setattr(img_libs_pkgs, "find_spec", fake_find_spec)
    _clear_caches()

    assert "my_backend" in img_libs_pkgs.get_img_lib_available()
    assert read_img.read_img["my_backend"]("x") == "r"
    assert read_img.read_img_pil["my_backend"]("x") == "p"
    assert read_img.read_img_ndarray["my_backend"]("x") == "n"


def test_plugin_version_unknown_when_distribution_missing(monkeypatch):
    from imgread_benchmark.img_libs import img_libs_pkgs
    from imgread_benchmark import read_img

    class FakeEntryPoint:
        name = "no_dist_backend"
        value = "pkgmod.sub:adapter"
        module = "pkgmod.sub"
        dist = None

        def load(self):
            return types.SimpleNamespace(read_img=lambda _path: "ok")

    monkeypatch.setattr(
        img_libs_pkgs, "entry_points", lambda **kwargs: [FakeEntryPoint()]
    )
    monkeypatch.setattr(img_libs_pkgs, "find_spec", lambda _name: object())

    def _missing(_pkg):
        raise PackageNotFoundError

    monkeypatch.setattr(read_img, "pkg_version", _missing)
    _clear_caches()

    versions = read_img.get_read_img_version()
    assert versions["no_dist_backend"] == "unknown"
