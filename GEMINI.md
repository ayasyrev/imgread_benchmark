# imgread_benchmark

A Python benchmarking tool for comparing the performance of various image reading libraries (Pillow, OpenCV, scikit-image, imageio, etc.).

## Project Overview

- **Language:** Python 3.12+
- **Build System:** `uv_build`
- **Package Manager:** `uv`
- **Execution:** ALWAYS use `uv run` for executing python and running tests (e.g., `uv run python ...`, `uv run pytest ...`).
- **Testing Framework:** `pytest`, `nox`
- **Core Dependencies:** `benchmark_utils`, `argparsecfg`, `rich`, `pillow`, `numpy`

## Getting Started

### Installation

This project uses `uv` for dependency management.

```bash
# Sync dependencies and create virtual environment
uv sync --extra img_libs
# Note: The new libraries (ajpegli, imagecodecs, simplejpeg, turbojpeg) are included in img_libs.
# Their wheels usually include necessary native libraries, avoiding extra system dependencies.
```

### Running the Application

The project provides a unified CLI with multiple subcommands:

**Primary usage - Benchmark:**
```bash
uv run imgread_benchmark <img_path> [options]
```

**Subcommands:**
- `uv run imgread_benchmark libs` - List available/installed image libraries
- `uv run imgread_benchmark data <dataset> [--size SIZE]` - Download benchmark datasets
- `uv run imgread_benchmark --version` - Show version

**Or via python module:**
```bash
uv run python -m imgread_benchmark <img_path> [options]
```

## Development Workflow

### Testing

**Daily Development:**
Run tests directly using `pytest` via `uv` for fast feedback.

```bash
uv run pytest
```

**Pre-release Verification:**
Use `nox` to run tests across all supported Python versions (3.12 - 3.13) before releasing.

```bash
# Run all sessions
nox
```

### Benchmarking Logic

The core logic resides in `src/imgread_benchmark/benchmark.py`.
- `BenchmarkImgRead`: Main class inheriting from `benchmark_utils.BenchmarkIter`.
- `READ_TO_FORMAT`: Maps target formats (`def` (default), `pil`, `np`) to their respective reading functions.

### Directory Structure

- `src/imgread_benchmark/`: Source code.
  - `img_libs/`: Wrappers/adapters for different image libraries (cv2, PIL, etc.).
  - `cli.py`: Unified CLI entry point with subcommands (benchmark, libs, data).
  - `cl_app.py`: Legacy benchmark CLI (deprecated, use `cli.py` instead).
  - `cl_versions.py`: Legacy versions CLI (deprecated, use `cli.py` instead).
  - `tests/`: Pytest suite.
  - `noxfile.py`: Automation definitions for testing.

## Key Configuration Files

- `pyproject.toml`: Project metadata, dependencies, and tool configurations (ruff, mypy, pytest).
- `uv.lock`: Locked dependency versions.
- `Makefile`: Shortcuts for building and distributing (e.g., `make pypi`, `make dist`).
