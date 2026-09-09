"""Explicit imgread paths; decoder errors propagate without a fallback reader."""

import imgread


def read(path):
    return imgread.load_numpy(path, color="rgb", dtype="uint8", backend="auto")


def read_bytes(data):
    return imgread.load_numpy_from_bytes(
        data, color="rgb", dtype="uint8", backend="auto"
    )


def persistent_reader():
    return imgread.Loader(color="rgb", dtype="uint8", backend="auto")


def persistent_reader_bytes():
    """Keep the Loader alive through its bound method, reused by the Dataset."""
    loader = persistent_reader()
    decode = getattr(loader, "decode", None)
    if not callable(decode):
        raise ValueError(
            "imgread missing required API Loader.decode for storage='memory'; "
            "install a build providing it"
        )
    return decode
