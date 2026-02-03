# Repository Guidelines

## Project Structure & Module Organization
- `src/imgread_benchmark/` contains the core library and unified CLI (`cli.py`).
- `src/imgread_benchmark/img_libs/` holds adapters for different image-reading backends.
- `tests/` contains pytest tests; sample images live under `tests/test_imgs/`.
- `dist/` is build output (created by `python -m build` or `make dist`).

## Build, Test, and Development Commands
**ALWAYS** use `uv` to run python scripts, commands and manage dependencies.
- `uv run python -m build` or `make dist`: build source and wheel distributions into `dist/`.
- `make clean`: remove `dist/` build artifacts.
- `uv run pytest`: run tests locally using your current environment.
- `uv run pytest --cov`: run tests with coverage reporting.
- `uv run ruff check .` or `uv run flake8 .`: run linters across the repo.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard PEP 8 naming (`snake_case` for modules/functions, `CamelCase` for classes).
- Format with Black (default settings) when in doubt; sort imports with isort.
- Lint with Ruff or Flake8; type-checking is done via mypy when needed.

## Testing Guidelines
- Tests are written with pytest and live in `tests/`.
- Test files follow `test_*.py`, and test functions/classes should follow pytest discovery rules.
- Use `uv run pytest --cov` for coverage reporting; add tests for new image readers or CLI behaviors.
- Some image libraries rely on optional system deps (e.g., `jpeg4py` needs `libjpeg-turbo`). If a system library is missing, related tests are skipped via runtime checks.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative; optional type prefixes like `build:` or `fix:` are used in history.
- PRs should describe the change, include rationale, and note how to test (commands + platform).
- If you modify benchmark outputs or sample data, mention affected files and expected deltas.

## Configuration Tips
- Optional image libraries are declared in `pyproject.toml` under the `img_libs` dependency group.
- CLI entry point is `imgread_benchmark` with subcommands: default (benchmark), `libs`, and `data`.
