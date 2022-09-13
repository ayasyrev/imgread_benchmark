from pathlib import Path

import typer
from benchmark_utils.benchmark import BenchmarkIter

from .get_img_filenames import get_img_filenames
from .read_image import read_img, read_img_pil, read_img_nparray
from .version import __version__


read_to_format ={
    "def": read_img,
    "pil": read_img_pil,
    "np": read_img_nparray,
}


app = typer.Typer()


@app.command()
def benchmark(
    img_path: Path = typer.Argument(..., help="Directory with images for benchmark"),
    num_samples: int = typer.Option(
        200, "-n", "--num_samples", help="Number of samples for test, default 200."
    ),
    to_format: str = typer.Option(
        "def", "-t", "--to", help="Format for read image to: default, Pil or Numpy."
    ),
    all: bool = typer.Option(False, "-A", "--all", help="Use all images from folder"),
    img_lib: str = typer.Option(None, "-l", "--img_lib", help="Image lib to test")
) -> None:
    if not img_path.exists():
        typer.echo(f"Img dir {img_path} dos not exist!")
        raise typer.Exit()
    if all:
        num_samples = 0
    filenames = get_img_filenames(img_path, num_samples=num_samples)
    if len(filenames) < num_samples:
        typer.echo(
            f"! Number of files in {img_path}: {len(filenames)} less than num_samples: {num_samples}"
        )

    read_fn_dict = read_to_format[to_format]
    if img_lib is not None:
        read_fn_dict = {img_lib: read_fn_dict[img_lib]}
    print("Benchmarking with images from {img_path}")
    if num_samples:
        print(f"number of samples: {num_samples}")
    else:
        print(f"{len(filenames)} images.")

    bench = BenchmarkIter(read_fn_dict, filenames, clear_progress=False)
    bench()


if __name__ == "__main__":  # pragma: no cover
    app()
