from importlib.metadata import PackageNotFoundError, version as pkg_version


try:
    __version__ = pkg_version("imgread_benchmark")
except PackageNotFoundError:
    __version__ = "0.0.0"
