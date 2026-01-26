# UV Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate packaging to `pyproject.toml` with uv dependency groups, update Python support to 3.10–3.14, and remove legacy requirements/setup files.

**Architecture:** Use PEP 621 metadata in `pyproject.toml`, keep core dependencies in `project.dependencies`, and move dev/test/img-libs to `tool.uv.dependency-groups`. Update nox sessions to install dependencies explicitly and test against 3.10–3.14.

**Tech Stack:** Python, setuptools, uv, nox.

---

### Task 1: Create `pyproject.toml` with PEP 621 metadata

**Files:**
- Create: `pyproject.toml`
- Remove: `setup.cfg`
- Remove: `setup.py`

**Step 1: Draft pyproject header and build-system**

Create `pyproject.toml` with:
- `[build-system]` using `setuptools` and `wheel`
- `[project]` metadata mapped from `setup.cfg`

**Step 2: Add project dependencies and scripts**

Populate:
- `project.dependencies` with packages from `requirements.txt`
- `[project.scripts]` for `imgread_benchmark` and `imgread_libs`
- `requires-python = ">=3.10"`

**Step 3: Add setuptools config for packages and version**

Add:
- `[tool.setuptools]` and `[tool.setuptools.packages.find]` for `src`
- `[tool.setuptools.dynamic]` version via `imgread_benchmark.version.__version__`

**Step 4: Remove legacy setup files**

Delete `setup.cfg` and `setup.py`.

**Step 5: Sanity check pyproject formatting**

Run: `python -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text())"`
Expected: no output and exit code 0.

---

### Task 2: Move dependency files into uv groups

**Files:**
- Modify: `pyproject.toml`
- Remove: `requirements.txt`
- Remove: `requirements_dev.txt`
- Remove: `requirements_test.txt`
- Remove: `requirements_img_libs.txt`
- Remove: `requirements_img_libs_conda.txt`

**Step 1: Add uv dependency groups**

Add `[tool.uv.dependency-groups]` with:
- `dev` from `requirements_dev.txt`
- `test` from `requirements_test.txt`
- `img_libs` from `requirements_img_libs.txt`
- `img_libs_conda` from `requirements_img_libs_conda.txt`

**Step 2: Remove legacy requirements files**

Delete all `requirements*.txt` files after porting contents.

**Step 3: Confirm uv groups are parseable**

Run: `uv pip compile --group dev --group test --dry-run`
Expected: completes without errors.

---

### Task 3: Update nox sessions for new Python versions and deps

**Files:**
- Modify: `noxfile.py`
- Modify: `noxfile_img_libs.py`
- Modify: `noxfile_lint.py`
- Modify: `noxfile_conda.py`
- Modify: `noxfile_conda_img_libs.py`
- Modify: `noxfile_conda_lint.py`

**Step 1: Update Python version lists**

Change all session python lists to `3.10`–`3.14`.

**Step 2: Replace extras installs with explicit deps**

For uv venvs, replace `-e .[test]` with:
- `-e .`
- `pytest`, `pytest-cov`, and any other explicit deps used by the session

**Step 3: Replace requirements file installs**

For img libs sessions, install the packages directly (e.g., `imageio`, `jpeg4py`, `kornia`, `kornia_rs`, `opencv-python-headless`, `scikit-image`).

**Step 4: Replace conda requirements file**

In conda img libs session, replace `requirements_img_libs_conda.txt` usage with explicit `conda_install("accimage", "imread")`.

---

### Task 4: Generate uv lockfile

**Files:**
- Create: `uv.lock`

**Step 1: Create lockfile**

Run: `uv lock`
Expected: `uv.lock` created/updated.

---

### Task 5: Verify tests and metadata

**Files:**
- Modify: `pyproject.toml` (if metadata fixes needed)

**Step 1: Run unit tests on one version**

Run: `nox -s tests-3.10`
Expected: tests pass.

**Step 2: (Optional) Spot-check another version**

Run: `nox -s tests-3.14`
Expected: tests pass.

---

### Task 6: Cleanup and summary

**Files:**
- Modify: `README.md` (only if we should document uv usage)

**Step 1: Decide whether to mention uv in README**

If desired, add minimal uv usage instructions.

**Step 2: Summarize changes for review**

List key file removals and new files for reviewers.
