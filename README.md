# imgread_benchmark
Benchmark for read images with different libs.

## List Available Image Libraries

```bash
UV_CACHE_DIR=.uv-cache uv run imgread_benchmark libs
```

## Benchmarking

To run the benchmark against a directory of images:

```bash
uv run imgread_benchmark /path/to/images [options]
```

### Options

- `-n`, `--num_samples`: Number of images to use for the benchmark (default: 200).
- `-t`, `--to`: Target format (`def`, `pil`, `np`).
- `-r`, `--repeats`: Number of passes over the selected images for each library (default: 5).
- `--shuffle`: Shuffle the selected files before every repeat of each library, including multiprocessing runs (default: off).
- `--no-warmup`: Disable file-cache warmup (enabled by default).
- `-l`, `--img_lib`: Test only a specific library.
- `-x`, `--exclude`: Exclude a specific library from the test.
- `-m`, `--multiprocessing`: Use multiprocessing for benchmarking.
- `-A`, `--all`: Use all images from the folder (ignores `-n`).

Example:
```bash
uv run imgread_benchmark /path/to/images -r 10 -n 100 -t np --shuffle
```

Before the first timed benchmark, all selected files are read once in chunks to
warm the OS file cache, without decoding or keeping the dataset in Python memory.
Warmup and shuffling are excluded from the timings. Shuffling changes only the
order, keeping the same selected images for every library and repeat.

## Dataset Management

You can download standard datasets for benchmarking using the `imgread_benchmark data` subcommand.

### Imagenette

To download the full Imagenette dataset:
```bash
uv run imgread_benchmark data imagenette
```

To download a specific size (e.g., 160px or 320px):
```bash
uv run imgread_benchmark data imagenette --size 160
```

Datasets are stored in the `.data/` directory by default.

## Add External Backend Plugin (No Repo Code Changes)

`imgread_benchmark` discovers external backends via Python entry points.
You do not need to modify this repository to add your backend.

1. Create a plugin package (local or published) with an adapter module.
2. Expose your backend via entry point group `imgread_benchmark.img_libs`.
3. Install the plugin package into the same environment as `imgread_benchmark`.
4. Verify with `imgread_benchmark libs`.

Minimal adapter example (`my_backend_plugin.py`):

```python
from PIL import Image
import numpy as np


def read_img(path: str):
    return read_img_ndarray(path)


def read_img_pil(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def read_img_ndarray(path: str) -> np.ndarray:
    return np.asarray(read_img_pil(path))


def is_available() -> bool:
    return True
```

Plugin `pyproject.toml`:

```toml
[project]
name = "my-backend-plugin"
version = "0.1.0"
dependencies = ["imgread_benchmark", "pillow", "numpy"]

[project.entry-points."imgread_benchmark.img_libs"]
my_backend = "my_backend_plugin"
```

Install plugin into current environment:

```bash
UV_CACHE_DIR=.uv-cache uv pip install -e /path/to/my-backend-plugin
```

Verify discovery and run benchmark:

```bash
UV_CACHE_DIR=.uv-cache uv run imgread_benchmark libs
UV_CACHE_DIR=.uv-cache uv run imgread_benchmark /path/to/images -l my_backend
```

Notes:

- `read_img`, `read_img_pil`, and `read_img_ndarray` are optional individually.
- For `-t def`, backend should provide `read_img`.
- For `-t pil`, backend should provide `read_img_pil`.
- For `-t np`, backend should provide `read_img_ndarray`.
- If plugin name conflicts with a built-in backend name, built-in backend wins.
- Built-in backends are listed first; additional and plugin backends are appended.

## Install Local Rust/PyO3 Backend

Example installing a local backend crate into this project environment:

```bash
env -u CONDA_PREFIX \
  PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
  UV_CACHE_DIR=.uv-cache \
  /path/to/maturin develop --manifest-path /path/to/backend/Cargo.toml --uv
```

Then verify:

```bash
UV_CACHE_DIR=.uv-cache uv run imgread_benchmark libs
```

### CPU PyTorch DataLoader benchmark

Compare Pillow, torchvision, two explicit OpenCV RGB paths and optional imgread with a common
uint8 resize/crop pipeline and real CPU DataLoader batches:

```bash
uv sync --extra dataloader --extra monitor
uv run --extra dataloader imgread_benchmark dataloader --list-readers
uv run --extra dataloader imgread_benchmark dataloader tests/test_imgs --reader pil-rgb --num-workers 0 --batch-size 2 --epochs 3 --output /tmp/imgread-dataloader-example-a
```

Add `--monitor-resources` for external CPU/RSS observations and a prepared-process
memory baseline. Use a saved `--manifest` to compare readers on the same ordered
file list. The new `--num-workers 0` means loading in the isolated consumer;
legacy `benchmark --nw 0` still means all CPUs. Optional dependencies preserve the
base installation. Supported execution: Linux, Python 3.12–3.13, torch 2.10.0 and
torchvision 0.25.0. Run `uv run --extra dataloader imgread_benchmark dataloader --help`
for the available options.

Use `--storage memory` to preload compressed bytes before timing and decode each
sample from RAM in the same Dataset. Add `--no-geometry` to omit resize/crop;
native sizes may require `--batch-size 1`. `--reader imgread-rgb` supports both
files and memory; `--reader imgread-loader-rgb` reuses one `imgread.Loader` per
process, calling the Loader for files and `Loader.decode(data)` for memory.
Memory mode requires an imgread build providing `Loader.decode`; PyPI 0.2.0
does not include it. Install a wheel from a compatible imgread build into the
benchmark environment. With worker processes, use `--persistent-workers` to
retain their Loader state between epochs. The encoded cache contains immutable
`bytes`; each access decodes a fresh RGB uint8 image.

With that build installed, use `--no-sync` to keep uv from replacing it with the
version pinned in `uv.lock`:

```bash
uv run --no-sync imgread_benchmark dataloader --list-readers --storage memory
uv run --no-sync imgread_benchmark dataloader /path/to/images --reader imgread-loader-rgb --storage memory --num-workers 2 --persistent-workers --epochs 3 --output /tmp/imgread-loader-memory
```

Both the `imgread` and `img_libs` extras install `imgread>=0.2.0` and register
it with the ordinary file benchmark (`-l imgread`). The published 0.2.0 release
supports the functional `imgread-rgb` reader with files and encoded bytes.
The persistent reader checks for the required Loader API before starting a run.
