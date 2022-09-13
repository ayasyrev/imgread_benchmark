from pathlib import Path
import typer

from .version import __version__
from .read_image import img_libs


app = typer.Typer()


@app.command()
def imgread(
    version: bool = typer.Option(
        False, "-v", "--version", help="App version and img libs available."),
) -> None:
    print("imgread.")
    if version:
        print(f"version: {__version__}")
        print("Availible image libs:")
        max_len = max(len(lib_name) for lib_name in img_libs)
        for img_lib in img_libs:
            print(f"    {img_lib:{max_len}} {img_libs[img_lib].__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
