# Lazy Imports Implementation Plan

## Problem Summary

**Current Issue:** Running `imgread_data imagenette` (which only needs `datasets.py`) triggers the package `__init__.py`, which imports benchmarking modules, causing jpeg4py to be imported and tested, even though it's not needed for data downloading.

**Error:**
```
Exception ignored in: <function JPEG.__del__ at 0x7f1fa189ee80>
Traceback (most recent call last):
  File "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.12/site-packages/jpeg4py/_py.py", line 215, in __del__
    if self.decompressor is not None:
       ^^^^^^^^^^^^^^^^^
AttributeError: 'JPEG' object has no attribute 'decompressor'
```

**Root Cause Chain:**
```
imgread_data
  → imports imgread_benchmark.cl_data
  → Python loads imgread_benchmark package
  → __init__.py imports: BenchmarkImgRead, get_img_filenames, img_lib_available
  → img_lib_available triggers _build_img_lib_available()
  → tests jpeg4py by importing and creating JPEG object
  → jpeg4py has __del__ bug → AttributeError
```

## Solution Strategy

Implement lazy imports at two levels:
1. **Package level** (`__init__.py`): Use lazy imports for heavy modules
2. **Module level** (`read_img.py`, `img_libs_pkgs.py`): Defer library detection until actually needed

**Goal:** `imgread_data` command should run without triggering any image library imports.

---

## Phase 1: Fix Package `__init__.py`

**File:** `src/imgread_benchmark/__init__.py`

**Approach:** Replace direct imports with lazy import mechanism using `__getattr__`.

**Current Code:**
```python
from .benchmark import BenchmarkImgRead
from .get_img_filenames import get_img_filenames
from .img_libs.img_libs_pkgs import img_lib_available


__all__ = ["get_img_filenames", "BenchmarkImgRead", "img_lib_available"]
```

**New Code:**
```python
def __getattr__(name: str):
    """Lazy import heavy modules only when accessed."""
    if name == "BenchmarkImgRead":
        from .benchmark import BenchmarkImgRead
        return BenchmarkImgRead
    elif name == "get_img_filenames":
        from .get_img_filenames import get_img_filenames
        return get_img_filenames
    elif name == "img_lib_available":
        from .img_libs.img_libs_pkgs import img_lib_available
        return img_lib_available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_img_filenames", "BenchmarkImgRead", "img_lib_available"]
```

**Benefits:**
- Only loads modules when accessed
- `imgread_data` won't trigger benchmarking imports
- Maintains API compatibility
- Python 3.7+ standard pattern

---

## Phase 2: Lazy Image Library Detection

**File:** `src/imgread_benchmark/img_libs/img_libs_pkgs.py`

**Current Problem:** `img_lib_available` is computed at module import time (line 60), triggering jpeg4py test.

**Solution:** Convert to cached lazy function.

**Current Code (Lines 49-60):**
```python
def _build_img_lib_available() -> list[str]:
    available: list[str] = []
    for lib_name in lib_to_package:
        if find_spec(lib_name) is None:
            continue
        if lib_name == "jpeg4py" and not _is_jpeg4py_usable():
            continue
        available.append(lib_name)
    return available


img_lib_available = _build_img_lib_available()
```

**New Code:**
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_img_lib_available() -> list[str]:
    """Get list of available image libraries (lazy, cached)."""
    available: list[str] = []
    for lib_name in lib_to_package:
        if find_spec(lib_name) is None:
            continue
        if lib_name == "jpeg4py" and not _is_jpeg4py_usable():
            continue
        available.append(lib_name)
    return available


# For backwards compatibility - deprecated, use get_img_lib_available()
img_lib_available = get_img_lib_available()
```

---

## Phase 3: Update `read_img.py`

**File:** `src/imgread_benchmark/read_img.py`

**Current Problem:** Module-level code (lines 43-47) loads all image libraries at import time.

**Current Code (Lines 14, 43-47):**
```python
from .img_libs.img_libs_pkgs import img_lib_available, lib_to_package

# ... lines 14-42 ...

img_libs = {}


for img_lib in img_lib_available:
    img_libs[img_lib] = load_lib(img_lib)
```

**New Code:**
```python
from .img_libs.img_libs_pkgs import lib_to_package, get_img_lib_available

# ... lines 14-42 ...

@lru_cache(maxsize=1)
def get_img_libs() -> dict:
    """Get loaded image libraries (lazy, cached)."""
    return {lib: load_lib(lib) for lib in get_img_lib_available()}


# For backwards compatibility
img_libs = get_img_libs()
```

**Also update the function dictionaries (lines 50-70):**

**Current:**
```python
def get_func_dict(
    func_name: str, func_dict: dict[str, Any]
) -> dict[str, Callable[[str], Any]]:
    """Return dict lib_name: func for given func_name"""
    return {
        lib_name: graceful_degradation(func)
        for lib_name in img_lib_available
        if (func := getattr(func_dict[lib_name], func_name, None)) is not None
    }


read_img: Dict[str, Callable[[str], Any]] = get_func_dict("read_img", img_libs)
read_img_pil: Dict[str, Callable[[str], Image.Image]] = get_func_dict(
    "read_img_pil", img_libs
)
read_img_ndarray: Dict[str, Callable[[str], ndarray]] = get_func_dict(
    "read_img_ndarray", img_libs
)
read_img_version: Dict[str, str] = {
    lib_name: pkg_version(lib_to_package[lib_name]) for lib_name in img_lib_available
}
```

**New:**
```python
def get_func_dict(
    func_name: str, func_dict: dict[str, Any]
) -> dict[str, Callable[[str], Any]]:
    """Return dict lib_name: func for given func_name"""
    return {
        lib_name: graceful_degradation(func)
        for lib_name in get_img_lib_available()
        if (func := getattr(func_dict[lib_name], func_name, None)) is not None
    }


@lru_cache(maxsize=1)
def get_read_img() -> Dict[str, Callable[[str], Any]]:
    """Get read_img function dict (lazy, cached)."""
    return get_func_dict("read_img", get_img_libs())


@lru_cache(maxsize=1)
def get_read_img_pil() -> Dict[str, Callable[[str], Image.Image]]:
    """Get read_img_pil function dict (lazy, cached)."""
    return get_func_dict("read_img_pil", get_img_libs())


@lru_cache(maxsize=1)
def get_read_img_ndarray() -> Dict[str, Callable[[str], ndarray]]:
    """Get read_img_ndarray function dict (lazy, cached)."""
    return get_func_dict("read_img_ndarray", get_img_libs())


@lru_cache(maxsize=1)
def get_read_img_version() -> Dict[str, str]:
    """Get version dict for image libraries (lazy, cached)."""
    return {
        lib_name: pkg_version(lib_to_package[lib_name])
        for lib_name in get_img_lib_available()
    }


# For backwards compatibility
read_img = get_read_img()
read_img_pil = get_read_img_pil()
read_img_ndarray = get_read_img_ndarray()
read_img_version = get_read_img_version()
```

---

## Phase 4: Update Dependent Files

### File: `src/imgread_benchmark/cl_versions.py`

**Current (Line 5):**
```python
from .read_img import img_libs, read_img_version
```

**New:**
```python
from .read_img import get_img_libs, get_read_img_version

# Use the lazy functions
img_libs = get_img_libs()
read_img_version = get_read_img_version()
```

**Update function usage (lines 24-31):**
```python
def imgread_versions(
    cfg: AppConfig,
) -> None:
    """Imgread_benchmark. Helpers utils for check and benchmark libs for read image files."""
    print("Imgread. Helpers utils for check and benchmark libs for read image files.")
    img_libs = get_img_libs()  # Changed from module-level
    versions = get_read_img_version()  # Changed from module-level
    print(f"Available {len(img_libs)} image libs:")

    if cfg.version:
        print(f"version: {__version__}")
    else:
        max_len = max(len(lib_name) for lib_name in img_libs)
        for img_lib in img_libs:
            print(f"    {img_lib:{max_len}} {versions[img_lib]}")  # Use versions dict
```

---

## Phase 5: Improve jpeg4py Error Handling

**File:** `src/imgread_benchmark/img_libs/img_libs_pkgs.py`

**Current `_is_jpeg4py_usable()` (Lines 19-46):**
Has overly broad `except Exception:` that catches everything.

**Improved Version:**
```python
def _is_jpeg4py_usable() -> bool:
    """Test if jpeg4py is working (has known __del__ bug)."""
    try:
        import jpeg4py
        from pathlib import Path
        import tempfile

        minimal_jpeg = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00\x43\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01"
            b"\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01"
            b"\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01"
            b"\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01"
            b"\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0"
            b'\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff'
            b"\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xda\x00"
            b"\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfd\x9f\xff\xd9"
        )

        tmp_path = None
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(minimal_jpeg)
            tmp_path = tmp_file.name

        try:
            result = jpeg4py.JPEG(tmp_path).decode()
            # Explicitly delete to trigger __del__ and catch the bug immediately
            del result
            return True
        except (AttributeError, OSError, ValueError) as e:
            # Known jpeg4py __del__ bug or other errors
            error_str = str(e)
            if "decompressor" in error_str:
                # Known bug in jpeg4py's __del__ method
                return False
            # Other errors, log and skip
            import warnings
            warnings.warn(f"jpeg4py test failed: {e}", RuntimeWarning)
            return False
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        # Import failed or other critical error
        import warnings
        warnings.warn(f"jpeg4py not available: {e}", RuntimeWarning)
        return False
```

---

## Phase 6: Test Coverage

### New Test File: `tests/test_lazy_imports.py`

```python
"""Test that data loading doesn't import image libraries."""

import sys


def test_data_download_no_img_libs_import(monkeypatch):
    """Test that imgread_data doesn't import image libraries."""
    # Track if jpeg4py was imported
    jpeg4py_imported = [False]

    original_import = __builtins__.__import__
    def mock_import(name, *args, **kwargs):
        if name == 'jpeg4py' or name.startswith('jpeg4py.'):
            jpeg4py_imported[0] = True
            raise ImportError("jpeg4py blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(__builtins__, '__import__', mock_import)

    # Import and use data download module
    from imgread_benchmark.cl_data import download_app, DATASET_PROVIDERS

    # Should not have imported jpeg4py
    assert not jpeg4py_imported[0], "jpeg4py should not be imported for data download"

    # Verify datasets are accessible
    assert "imagenette" in DATASET_PROVIDERS


def test_lazy_img_lib_detection():
    """Test that image library detection is lazy."""
    # This should not import jpeg4py
    from imgread_benchmark.img_libs.img_libs_pkgs import get_img_lib_available

    # First call should work
    libs = get_img_lib_available()
    assert isinstance(libs, list)

    # Second call should be cached
    libs2 = get_img_lib_available()
    assert libs is libs2  # Same object (cached)


def test_package_lazy_imports():
    """Test that package imports are lazy."""
    import imgread_benchmark

    # These should not trigger benchmarking imports yet
    assert hasattr(imgread_benchmark, '__getattr__')

    # Accessing BenchmarkImgRead should trigger import
    from imgread_benchmark import BenchmarkImgRead
    assert BenchmarkImgRead is not None
```

### Update Existing Tests

**File: `tests/test_img_libs_pkgs.py`**

Add tests for new function-based API:
```python
def test_get_img_lib_available():
    """Test get_img_lib_available function."""
    from imgread_benchmark.img_libs.img_libs_pkgs import get_img_lib_available

    libs = get_img_lib_available()
    assert isinstance(libs, list)
    # PIL should always be available (it's a required dependency)
    assert "PIL" in libs


def test_get_img_lib_available_cached():
    """Test that get_img_lib_available is cached."""
    from imgread_benchmark.img_libs.img_libs_pkgs import get_img_lib_available

    libs1 = get_img_lib_available()
    libs2 = get_img_lib_available()
    assert libs1 is libs2
```

**File: `tests/test_datasets.py`**

Verify datasets work independently:
```python
def test_datasets_without_img_libs(monkeypatch):
    """Test that dataset providers work without image libraries."""
    # Block all image library imports
    def mock_import(name, *args, **kwargs):
        if name in ['jpeg4py', 'cv2', 'skimage', 'imageio']:
            raise ImportError(f"{name} blocked for test")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr(__builtins__, '__import__', mock_import)

    from imgread_benchmark.datasets import DatasetProvider, ImagenetteProvider

    provider = ImagenetteProvider()
    assert provider.name == "imagenette"
    assert provider.default_size == "full"

    url = provider.get_url("160")
    assert "imagenette2-160" in url
```

---

## Implementation Order

Execute phases in this order, testing after each phase:

1. **Phase 5** - Improve jpeg4py error handling (independent, safer)
2. **Phase 1** - Fix `__init__.py` lazy imports (highest impact)
3. **Phase 2** - Lazy library detection
4. **Phase 3** - Update `read_img.py`
5. **Phase 4** - Update `cl_versions.py`
6. **Phase 6** - Add test coverage

### Test After Each Phase

```bash
# Test that data download works
uv run imgread_data imagenette --size 160

# Test that benchmarking still works
uv run imgread_libs

# Test all tests pass
uv run pytest tests/ -v

# Run linting
uv run ruff check .
```

---

## Backwards Compatibility

✅ **Fully Maintained:**
- Package API unchanged (`from imgread_benchmark import X` still works)
- All existing imports continue to work
- CLI commands unchanged
- Module-level variables still exist (backwards compatibility layer)

✅ **New API:**
- `get_img_lib_available()` - Function-based lazy version
- `get_img_libs()` - Function-based lazy version
- `get_read_img()`, `get_read_img_pil()`, etc. - Function-based lazy versions

⚠️ **Breaking changes:** None

---

## Success Criteria

After implementation:

1. ✅ `uv run imgread_data imagenette` runs without importing jpeg4py
2. ✅ No AttributeError from jpeg4py `__del__`
3. ✅ `uv run imgread_libs` still lists all available libraries
4. ✅ All existing tests pass
5. ✅ New tests verify lazy loading behavior
6. ✅ Code passes linting (`ruff check .`)
7. ✅ Backwards compatibility maintained

---

## Rollback Plan

If issues arise:

1. **Quick rollback:** Revert `__init__.py` changes (Phase 1) - immediately fixes the main issue
2. **Partial rollback:** Keep Phase 5 (better error handling) but revert other phases
3. **Full rollback:** Git revert to commit before changes

All changes are backwards compatible, so rolling back won't break existing code.

---

## Future Improvements

After successful implementation:

1. **Consider deprecating module-level variables** in favor of functions
2. **Add logging** to track when lazy imports are triggered
3. **Performance testing** to ensure caching is effective
4. **Documentation updates** to explain lazy import behavior
5. **Consider separating `imgread_data` into its own package** for even cleaner separation

---

## Notes

- **Python 3.7+ required** for `__getattr__` lazy import pattern (project requires 3.10+, so OK)
- **`functools.lru_cache`** used for caching lazy function results
- **Backwards compatibility layer** maintains existing module-level variables
- **All changes are additive** - no deletions, only additions and refactoring
- **Test coverage** ensures lazy loading works correctly

---

**Created:** 2025-02-02
**Status:** Ready for implementation
**Estimated time:** 2-3 hours for all phases
