# imgread_benchmark
Benchmark for read images with different libs.

Requires Python 3.12 or 3.13. The ordinary file benchmark is portable;
DataLoader execution and the encoded tmpfs cache are qualified on Linux.

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

### Decode encoded buffers from memory

`--decode-only` measures decoding through each library's buffer adapter:

```bash
uv run imgread_benchmark /path/to/images --decode-only -l PIL -t np -r 3
uv run imgread_benchmark /path/to/images --decode-only -l imgread -t np -m --nw 2
uv run imgread_benchmark --decode-only --cache-limit 4GiB /path/to/images -A
```

- `--cache-dir` defaults to `/dev/shm/imgread_benchmark-<uid>`. It must be on
  Linux tmpfs, owned by the current user, with private permissions (`0700`).
- `--cache-limit` limits the total allocated data/index space under that root
  (default `2GiB`; positive bytes, `KiB`, `MiB`, or `GiB`). Inactive caches are
  evicted by last use; active process leases prevent eviction.
- The immutable cache contains compressed source bytes in `data.bin` and a
  versioned `index.json`. Full and subset requests reuse a matching cache after
  checking source paths, sizes, and modification/change timestamps. A cache hit
  does not reread source contents. Cache files persist after the benchmark.
- New caches are copied incrementally and published atomically. Preparation
  checks tmpfs free space, available RAM and cgroup memory headroom, retaining
  a reserve of at least 512 MiB or 10% of the effective memory limit.
- Mapping, page prefault, a decoder initialization probe, and the default full
  decode warmup happen before timing. `--no-warmup` skips the full warmup only;
  `--shuffle` rearranges indices outside each timed repeat.
- Sequential timing includes decoder calls, buffer/format conversions and result
  disposal. Multiprocessing uses persistent spawn workers per backend, with an
  independent mapping and lease in each worker. Timed throughput includes task
  dispatch and status synchronization; image payloads are not sent through IPC.
- Output identifies preparation time, cache hit/build, path, size, evictions,
  selected count, workers, warmup/shuffle policy and decoder versions. Python
  users can inspect the same metadata and failures in `benchmark.decode_report`.
- A failed backend has no reported speed. Other backends may finish, but the CLI
  exits nonzero (the Python API raises `DecodeRunError`). Unsupported buffer
  adapters are listed as skipped; explicitly selecting one is an error.

No resize or normalization is added. Each backend keeps its file adapter's
`def`/`pil`/`np` semantics. Buffer copies required by a library count toward its
time; this mode does not promise zero-copy decoding or pinned RAM. tmpfs may swap.
Worker startup and lack of progress each have a 120-second bound, followed by
bounded graceful/terminate/kill cleanup (10/5/2 seconds).

```python
from imgread_benchmark import BenchmarkImgRead

bench = BenchmarkImgRead(filenames=selected_paths, target_format="np",
                         decode_only=True, cache_limit="2GiB", num_repeats=3)
bench.run(func_name=["PIL", "cv2", "imgread"], multiprocessing=True, num_workers=2)
print(bench.results)        # average seconds per complete pass
print(bench.decode_report)  # metadata, per-pass seconds and errors
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
- Decode-only plugins expose optional `decode_img`, `decode_img_pil` and
  `decode_img_ndarray`. They accept a contiguous read-only `memoryview`, return
  a decoded independent image and never retain the input or open a source path.
  Errors must propagate. Spawn mode requires importable, pickleable callables.

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
The published `imgread>=0.2.1` provides this API. With worker processes, use `--persistent-workers` to
retain their Loader state between epochs. The encoded cache contains immutable
`bytes`; each access decodes a fresh RGB uint8 image.

Install the optional dependencies and run:

```bash
uv sync --extra dataloader --extra imgread
uv run --extra dataloader --extra imgread imgread_benchmark dataloader --list-readers --storage memory
uv run --extra dataloader --extra imgread imgread_benchmark dataloader /path/to/images --reader imgread-loader-rgb --storage memory --num-workers 2 --persistent-workers --epochs 3 --output /tmp/imgread-loader-memory
```

Both the `imgread` and `img_libs` extras install `imgread>=0.2.1` and register
it with the ordinary file and decode-only benchmarks (`-l imgread`).
The persistent reader checks for the required Loader API before starting a run.

The Python `run_benchmark(..., timeout_seconds=...)` budget starts at function
entry and covers isolated dependency/manifest preflight, consumer preparation,
monitor startup and epochs. No consumer is started after the preflight budget
expires. Timeout raises `BenchmarkRunError` with a structured partial result and
the failing stage; invalid configuration still raises `ValueError`. Process
cleanup has a separate finite budget. Calls made earlier to `snapshot_files` or
`discover_manifest` are outside this budget. Calls from Python threads are supported.
