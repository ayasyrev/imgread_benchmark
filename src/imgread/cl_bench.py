from pathlib import Path

import typer

from .benchmark import BenchmarkImgRead
from .get_img_filenames import get_img_filenames

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
    img_lib: str = typer.Option(None, "-l", "--img_lib", help="Image lib to test"),
    exclude_lib: str = typer.Option(
        None, "-x", "--exclude", help="Image lib exclude from test"
    ),
) -> None:
    """Benchmark read image functions."""
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

    print(f"Benchmarking with images from {img_path}, convert to: {to_format}")
    if num_samples:
        print(f"number of samples: {num_samples}")
    else:
        print(f"{len(filenames)} images.")

    bench = BenchmarkImgRead(
        filenames=filenames,
        img_lib_name=img_lib,
        to_format=to_format,
        exclude_name=exclude_lib,
    )
    bench()


if __name__ == "__main__":  # pragma: no cover
    app()
