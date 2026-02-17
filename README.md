# imgread_benchmark
Benchmark for read images with different libs.

## List Available Image Libraries

```bash
UV_CACHE_DIR=.uv-cache uv run imgread_benchmark libs
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

## Plugin Backends (No Repo Changes Needed)

`imgread_benchmark` supports external image backends via Python entry points.
After you install a plugin package into the same environment, it is discovered
automatically by `imgread_benchmark libs`.

Entry point group:

- `imgread_benchmark.img_libs`

Example in plugin package `pyproject.toml`:

```toml
[project.entry-points."imgread_benchmark.img_libs"]
imgread_rs = "imgread_rs_plugin:adapter"
```

The entry point value should resolve to an adapter object/module that can expose
any of these callables (partial adapters are allowed):

- `read_img(path: str)`
- `read_img_pil(path: str)`
- `read_img_ndarray(path: str)`

Optional:

- `is_available() -> bool` for runtime checks (return `False` to hide backend)

Notes:

- If a plugin name conflicts with a built-in backend name, the built-in backend wins.
- Built-in backends are listed first; plugin backends are appended after discovery.

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
