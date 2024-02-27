import typer

from .read_img import img_libs
from .version import __version__

app = typer.Typer()


@app.command()
def imgread(
    version: bool = typer.Option(False, "-v", "--version", help="App version."),
) -> None:
    """Imgread. Helpers utils for check and benchmark libs for read image files."""
    print("Imgread. Helpers utils for check and benchmark libs for read image files.")
    print(f"Available {len(img_libs)} image libs:")
    max_len = max(len(lib_name) for lib_name in img_libs)
    for img_lib in img_libs:
        print(f"    {img_lib:{max_len}} {img_libs[img_lib].__version__}")
    if version:
        print(f"version: {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
