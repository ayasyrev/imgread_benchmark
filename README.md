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
- `-r`, `--repeats`: Number of repeat runs for each image (default: 5).
- `-l`, `--img_lib`: Test only a specific library.
- `-x`, `--exclude`: Exclude a specific library from the test.
- `-m`, `--multiprocessing`: Use multiprocessing for benchmarking.
- `-A`, `--all`: Use all images from the folder (ignores `-n`).

Example:
```bash
uv run imgread_benchmark /path/to/images -r 10 -n 100 -t np
```

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

Compare Pillow, torchvision and two explicit OpenCV RGB paths with a common
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
torchvision 0.25.0. See [the complete contract](docs/dataloader-benchmark.md) for
flags, Python API, source image restrictions, timing and resource limitations.
