import pytest
from imgread_benchmark.read_img import graceful_degradation, read_img


def test_graceful_degradation_nonexistent_file():
    # This should not raise an exception if graceful degradation is implemented
    # Currently it will likely raise FileNotFoundError or similar from the underlying libs
    lib_name = "PIL"
    if lib_name in read_img:
        try:
            read_img[lib_name]("nonexistent.jpg")
        except Exception as e:
            pytest.fail(f"read_img[{lib_name}] raised {type(e).__name__}: {e}")


def test_graceful_degradation_propagates_type_error():
    def broken_reader(_img_path: str):
        raise TypeError("bad decode")

    wrapped = graceful_degradation(broken_reader)

    with pytest.raises(TypeError):
        wrapped("/tmp/nowhere.jpg")
