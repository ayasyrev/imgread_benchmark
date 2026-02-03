"""Test that data loading doesn't import image libraries."""


def test_data_download_no_img_libs_import(monkeypatch):
    """Test that imgread_data doesn't import image libraries."""
    import builtins

    # Track if jpeg4py was imported
    jpeg4py_imported = [False]

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "jpeg4py" or name.startswith("jpeg4py."):
            jpeg4py_imported[0] = True
            raise ImportError("jpeg4py blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Import and use data download module
    from imgread_benchmark.cl_data import DATASET_PROVIDERS

    # Should not have imported jpeg4py
    assert not jpeg4py_imported[0], "jpeg4py should not be imported for data download"

    # Verify datasets are accessible
    assert "imagenette" in DATASET_PROVIDERS


def test_lazy_img_lib_detection():
    """Test that image library detection is lazy."""
    # This should not import jpeg4py
    from imgread_benchmark.img_libs.img_libs_pkgs import get_img_lib_available

    # First call should work
    libs = get_img_lib_available()
    assert isinstance(libs, list)

    # Second call should be cached
    libs2 = get_img_lib_available()
    assert libs is libs2  # Same object (cached)


def test_package_lazy_imports():
    """Test that package imports are lazy."""
    import imgread_benchmark

    # These should not trigger benchmarking imports yet
    assert hasattr(imgread_benchmark, "__getattr__")

    # Accessing BenchmarkImgRead should trigger import
    from imgread_benchmark import BenchmarkImgRead

    assert BenchmarkImgRead is not None


def test_lazy_mapping_is_mapping():
    import collections.abc

    from imgread_benchmark import read_img as read_img_mod

    assert isinstance(read_img_mod.img_libs, collections.abc.Mapping)
    assert isinstance(read_img_mod.read_img, collections.abc.Mapping)
    assert isinstance(read_img_mod.read_img_pil, collections.abc.Mapping)
    assert isinstance(read_img_mod.read_img_ndarray, collections.abc.Mapping)
    assert isinstance(read_img_mod.read_img_version, collections.abc.Mapping)
