def __getattr__(name: str):
    """Lazy import heavy modules only when accessed."""
    if name == "BenchmarkImgRead":
        from .benchmark import BenchmarkImgRead

        return BenchmarkImgRead
    elif name == "get_img_filenames":
        from .get_img_filenames import get_img_filenames

        return get_img_filenames
    elif name == "img_lib_available":
        from .img_libs.img_libs_pkgs import img_lib_available

        return img_lib_available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_img_filenames", "BenchmarkImgRead", "img_lib_available"]
