# Decode-Only Memmap Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить Linux-режим `--decode-only`, который до замера собирает выбранные сжатые изображения в переиспользуемый memmap-кэш на tmpfs и измеряет декодирование из буферов.

**Architecture:** Неизменяемый `data.bin` и версионированный `index.json` создаются атомарно и выдаются через контекстную аренду с read-only memory map. Отдельный строгий реестр буферных декодеров питает новый последовательный или multiprocessing-исполнитель; `BenchmarkImgRead` выбирает новый путь только при `decode_only=True`, сохраняя обычный файловый режим.

**Tech Stack:** Python 3.12–3.13, stdlib `mmap`/`fcntl`/`multiprocessing`, NumPy, Pillow, argparsecfg, Rich, pytest, Ruff, uv.

**Design:** `docs/superpowers/specs/2026-09-04-decode-only-memmap-cache-design.md`

---

## Карта файлов

- Create `src/imgread_benchmark/encoded_cache.py`: формат индекса, построение и поиск кэша, лимит, LRU, tmpfs/RAM-проверки, блокировки, read-only mapping и аренда.
- Create `src/imgread_benchmark/decode_benchmark.py`: строгие последовательные и multiprocessing-проходы с прогревом и корректной границей таймера.
- Modify `src/imgread_benchmark/read_img.py`: ленивые реестры `decode_img*` без graceful degradation.
- Modify eleven modules in `src/imgread_benchmark/img_libs/`: буферные функции для поддерживаемых библиотек.
- Modify `src/imgread_benchmark/benchmark.py`: переключение `BenchmarkImgRead` между файловым и decode-only исполнителями.
- Modify `src/imgread_benchmark/cli.py`: новые аргументы, валидация и сообщения об ошибках.
- Modify `README.md`: пользовательский интерфейс и ограничения режима.
- Create `tests/test_encoded_cache.py`, `tests/test_decode_registry.py`, `tests/test_decode_adapters.py`, `tests/test_decode_benchmark.py`, `tests/test_decode_multiprocessing.py`.
- Modify `tests/test_benchmark_lazy.py`, `tests/test_cli_unified.py`, `tests/test_lazy_imports.py`.

---

### Task 1: Базовый неизменяемый memmap-кэш

**Files:**
- Create: `src/imgread_benchmark/encoded_cache.py`
- Create: `tests/test_encoded_cache.py`

- [ ] **Step 1: Write failing tests for size parsing, atomic construction, duplicate ordering, hits, and subset reuse**

```python
# tests/test_encoded_cache.py
from pathlib import Path

import pytest

from imgread_benchmark.encoded_cache import (
    CACHE_FORMAT_VERSION,
    CacheOptions,
    EncodedCacheManager,
    parse_size,
)


def _image(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _manager(tmp_path: Path, limit: int = 16 * 1024) -> EncodedCacheManager:
    return EncodedCacheManager(
        CacheOptions(
            root=tmp_path / "cache",
            limit_bytes=limit,
            require_tmpfs=False,
            memory_probe=lambda: (1 << 40, 1 << 40),
        )
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("17", 17), ("2KiB", 2048), ("3MiB", 3 * 1024**2), ("2GiB", 2 * 1024**3)],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "0", "-1", "1.5GiB", "2GB"])
def test_parse_size_rejects_invalid_values(text):
    with pytest.raises(ValueError, match="positive integer"):
        parse_size(text)


def test_builds_mapping_and_preserves_duplicates(tmp_path):
    first = _image(tmp_path / "first.jpg", b"first")
    second = _image(tmp_path / "second.jpg", b"second-value")
    manager = _manager(tmp_path)

    with manager.acquire([first, second, first]) as lease:
        assert lease.hit is False
        assert lease.format_version == CACHE_FORMAT_VERSION
        assert [bytes(item) for item in lease.buffers] == [
            b"first",
            b"second-value",
            b"first",
        ]
        assert lease.data_bytes == len(b"firstsecond-value")
        assert lease.data_path.name == "data.bin"
        assert lease.index_path.name == "index.json"


def test_exact_hit_does_not_copy_source_contents(tmp_path, monkeypatch):
    first = _image(tmp_path / "first.jpg", b"first")
    manager = _manager(tmp_path)
    with manager.acquire([first]):
        pass

    def fail_copy(*_args, **_kwargs):
        raise AssertionError("cache hit copied source data")

    monkeypatch.setattr(manager, "_copy_source", fail_copy)
    with manager.acquire([first]) as lease:
        assert lease.hit is True
        assert bytes(lease.buffers[0]) == b"first"


def test_smaller_selection_reuses_smallest_superset(tmp_path):
    files = [_image(tmp_path / f"{index}.jpg", bytes([index])) for index in range(3)]
    manager = _manager(tmp_path)
    with manager.acquire(files) as large:
        large_id = large.cache_id

    with manager.acquire([files[1]]) as subset:
        assert subset.hit is True
        assert subset.cache_id == large_id
        assert bytes(subset.buffers[0]) == b"\x01"


def test_source_change_builds_new_cache(tmp_path):
    source = _image(tmp_path / "image.jpg", b"old")
    manager = _manager(tmp_path)
    with manager.acquire([source]) as old:
        old_id = old.cache_id
    source.write_bytes(b"new-content")
    with manager.acquire([source]) as new:
        assert new.hit is False
        assert new.cache_id != old_id
        assert bytes(new.buffers[0]) == b"new-content"


def test_failed_build_publishes_no_cache(tmp_path, monkeypatch):
    files = [
        _image(tmp_path / "first.jpg", b"first"),
        _image(tmp_path / "second.jpg", b"second"),
    ]
    manager = _manager(tmp_path)
    original = manager._copy_source
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        original(source, target)

    monkeypatch.setattr(manager, "_copy_source", fail_second)
    with pytest.raises(OSError, match="injected copy failure"):
        manager.acquire(files)
    assert not list(manager.options.root.glob("cache-*"))
    assert not list(manager.options.root.glob(".build-*"))
```

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_encoded_cache.py -q
```

Expected: collection fails with `ModuleNotFoundError: imgread_benchmark.encoded_cache`.

- [ ] **Step 3: Implement the cache model, serialization, build, lookup, lease, and cleanup order**

Create these public types and keep helpers private:

```python
# src/imgread_benchmark/encoded_cache.py
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import BinaryIO

CACHE_FORMAT_VERSION = 1
_SIZE_RE = re.compile(r"^(?P<number>[1-9][0-9]*)(?P<unit>KiB|MiB|GiB)?$")
_UNITS = {None: 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}


class EncodedCacheError(RuntimeError):
    """Decode-only cache cannot be prepared or used."""


def parse_size(value: str | int) -> int:
    if isinstance(value, int):
        if value > 0:
            return value
        raise ValueError("cache size must be a positive integer")
    match = _SIZE_RE.fullmatch(value)
    if match is None:
        raise ValueError("cache size must be a positive integer with optional KiB, MiB, or GiB")
    return int(match.group("number")) * _UNITS[match.group("unit")]


@dataclass(frozen=True)
class SourceRecord:
    path: str
    size: int
    mtime_ns: int
    ctime_ns: int
    offset: int = 0
    length: int = 0

    @classmethod
    def inspect(cls, path: Path) -> "SourceRecord":
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        if not resolved.is_file() or stat.st_size <= 0:
            raise EncodedCacheError(f"encoded image is not a non-empty file: {resolved}")
        return cls(str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def fingerprint(self) -> tuple[str, int, int, int]:
        return (self.path, self.size, self.mtime_ns, self.ctime_ns)


@dataclass(frozen=True)
class CacheManifest:
    version: int
    cache_id: str
    data_bytes: int
    entries: tuple[SourceRecord, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "cache_id": self.cache_id,
            "data_bytes": self.data_bytes,
            "entries": [entry.__dict__ for entry in self.entries],
        }

    @classmethod
    def from_path(cls, path: Path) -> "CacheManifest":
        value = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(SourceRecord(**item) for item in value["entries"])
        return cls(value["version"], value["cache_id"], value["data_bytes"], entries)


@dataclass(frozen=True)
class CacheOptions:
    root: Path
    limit_bytes: int = 2 * 1024**3
    require_tmpfs: bool = True
    memory_probe: Callable[[], tuple[int, int]] | None = None


class EncodedCacheLease(AbstractContextManager["EncodedCacheLease"]):
    def __init__(
        self,
        cache_dir: Path,
        manifest: CacheManifest,
        order: Sequence[str],
        lock_file: BinaryIO,
        *,
        hit: bool,
        evicted: Sequence[str],
        preparation_seconds: float,
    ) -> None:
        self.cache_id = manifest.cache_id
        self.format_version = manifest.version
        self.data_bytes = manifest.data_bytes
        self.data_path = cache_dir / "data.bin"
        self.index_path = cache_dir / "index.json"
        self.hit = hit
        self.evicted = tuple(evicted)
        self.preparation_seconds = preparation_seconds
        self._lock_file = lock_file
        self._mapping: mmap.mmap | None = None
        self._views: list[memoryview] = []
        entries = {entry.path: entry for entry in manifest.entries}
        self._ranges = tuple((entries[path].offset, entries[path].length) for path in order)

    def __enter__(self) -> "EncodedCacheLease":
        try:
            data_file = self.data_path.open("rb")
            try:
                self._mapping = mmap.mmap(data_file.fileno(), 0, access=mmap.ACCESS_READ)
            finally:
                data_file.close()
            self._views = [
                memoryview(self._mapping)[offset : offset + length]
                for offset, length in self._ranges
            ]
            return self
        except Exception:
            self._release()
            raise

    @property
    def buffers(self) -> tuple[memoryview, ...]:
        if self._mapping is None:
            raise EncodedCacheError("cache lease is not open")
        return tuple(self._views)

    def prefault(self) -> int:
        if self._mapping is None:
            raise EncodedCacheError("cache lease is not open")
        page = mmap.PAGESIZE
        checksum = 0
        for offset in range(0, len(self._mapping), page):
            checksum ^= self._mapping[offset]
        checksum ^= self._mapping[-1]
        return checksum

    def __exit__(self, *_exc: object) -> None:
        self._release()

    def _release(self) -> None:
        for view in self._views:
            view.release()
        self._views.clear()
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()


class EncodedCacheManager:
    def __init__(self, options: CacheOptions) -> None:
        self.options = options

    def acquire(self, filenames: Sequence[str | Path]) -> EncodedCacheLease:
        started = time.perf_counter()
        inspected = [SourceRecord.inspect(Path(name)) for name in filenames]
        unique = {entry.path: entry for entry in inspected}
        normalized = tuple(sorted(unique.values(), key=lambda entry: entry.path))
        cache_id = self._cache_id(normalized)
        root = self.options.root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        if self.options.require_tmpfs:
            self._require_tmpfs(root)
        (root / ".locks").mkdir(exist_ok=True)
        (root / ".usage").mkdir(exist_ok=True)
        with self._exclusive_root_lock():
            self._remove_incomplete_builds()
            selected = self._find_smallest_superset(normalized)
            hit = selected is not None
            evicted = self._enforce_budget(
                requested_bytes=0 if hit else sum(item.size for item in normalized),
                protected_ids={selected[0].cache_id} if selected else set(),
            )
            if selected is None:
                manifest, cache_dir = self._build(cache_id, normalized)
            else:
                manifest, cache_dir = selected
            self._touch_usage(manifest.cache_id)
            lock_file = self._lock_path(manifest.cache_id).open("a+b")
            fcntl.flock(lock_file, fcntl.LOCK_SH)
        return EncodedCacheLease(
            cache_dir,
            manifest,
            [entry.path for entry in inspected],
            lock_file,
            hit=hit,
            evicted=evicted,
            preparation_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _cache_id(entries: Sequence[SourceRecord]) -> str:
        payload = json.dumps(
            [CACHE_FORMAT_VERSION, [entry.fingerprint() for entry in entries]],
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _build(
        self, cache_id: str, entries: Sequence[SourceRecord]
    ) -> tuple[CacheManifest, Path]:
        root = self.options.root
        final_dir = root / f"cache-{cache_id}"
        build_dir = Path(tempfile.mkdtemp(prefix=".build-", dir=root))
        built: list[SourceRecord] = []
        offset = 0
        try:
            with (build_dir / "data.bin").open("wb") as target:
                for source in entries:
                    before = SourceRecord.inspect(Path(source.path))
                    self._copy_source(Path(source.path), target)
                    after = SourceRecord.inspect(Path(source.path))
                    if before.fingerprint() != after.fingerprint():
                        raise EncodedCacheError(f"source changed while caching: {source.path}")
                    built.append(SourceRecord(*before.fingerprint(), offset, before.size))
                    offset += before.size
                target.flush()
                os.fsync(target.fileno())
            manifest = CacheManifest(CACHE_FORMAT_VERSION, cache_id, offset, tuple(built))
            (build_dir / "index.json").write_text(
                json.dumps(manifest.to_json(), separators=(",", ":")), encoding="utf-8"
            )
            self._validate_cache(build_dir, manifest)
            build_dir.rename(final_dir)
            return manifest, final_dir
        except Exception:
            shutil.rmtree(build_dir, ignore_errors=True)
            raise

    @staticmethod
    def _copy_source(source: Path, target: BinaryIO) -> None:
        with source.open("rb") as stream:
            shutil.copyfileobj(stream, target, length=1024 * 1024)
```

Add the private methods used by the basic implementation. Task 2 replaces the
minimal budget and tmpfs checks with the complete policies:

```python
    @contextmanager
    def _exclusive_root_lock(self) -> Iterator[None]:
        lock = (self.options.root / ".root.lock").open("a+b")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _lock_path(self, cache_id: str) -> Path:
        return self.options.root / ".locks" / f"{cache_id}.lock"

    def _touch_usage(self, cache_id: str) -> None:
        (self.options.root / ".usage" / cache_id).touch()

    def _remove_incomplete_builds(self) -> None:
        for path in self.options.root.glob(".build-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def _require_tmpfs(self, path: Path) -> None:
        try:
            path.resolve().relative_to("/dev/shm")
        except ValueError as exc:
            raise EncodedCacheError(f"decode-only cache must be on tmpfs: {path}") from exc

    def _validate_cache(self, cache_dir: Path, manifest: CacheManifest) -> None:
        data_path = cache_dir / "data.bin"
        if manifest.version != CACHE_FORMAT_VERSION:
            raise EncodedCacheError("unsupported cache format")
        if not data_path.is_file() or data_path.stat().st_size != manifest.data_bytes:
            raise EncodedCacheError("cache data length does not match index")
        expected_offset = 0
        for entry in sorted(manifest.entries, key=lambda item: item.offset):
            if entry.offset != expected_offset or entry.length != entry.size:
                raise EncodedCacheError("cache ranges are invalid")
            expected_offset += entry.length
        if expected_offset != manifest.data_bytes:
            raise EncodedCacheError("cache ranges do not cover data")
        if self._cache_id(manifest.entries) != manifest.cache_id:
            raise EncodedCacheError("cache id does not match entries")

    def _find_smallest_superset(
        self, requested: Sequence[SourceRecord]
    ) -> tuple[CacheManifest, Path] | None:
        wanted = {entry.path: entry.fingerprint() for entry in requested}
        candidates = []
        for cache_dir in self.options.root.glob("cache-*"):
            try:
                manifest = CacheManifest.from_path(cache_dir / "index.json")
                self._validate_cache(cache_dir, manifest)
                available = {entry.path: entry.fingerprint() for entry in manifest.entries}
                if all(available.get(path) == fingerprint for path, fingerprint in wanted.items()):
                    candidates.append((manifest, cache_dir))
            except (EncodedCacheError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
        return min(candidates, key=lambda item: item[0].data_bytes, default=None)

    def _enforce_budget(
        self, requested_bytes: int, protected_ids: set[str]
    ) -> list[str]:
        return []
```

- [ ] **Step 4: Run the cache tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_encoded_cache.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run static checks for the new module and commit**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check src/imgread_benchmark/encoded_cache.py tests/test_encoded_cache.py
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_encoded_cache.py tests/test_get_img_filenames.py -q
git add src/imgread_benchmark/encoded_cache.py tests/test_encoded_cache.py
git commit -m "feat: add encoded image memmap cache"
```

Expected: Ruff and pytest pass; commit contains only the cache module and its tests.

---

### Task 2: Лимит, LRU, блокировки и восстановление

**Files:**
- Modify: `src/imgread_benchmark/encoded_cache.py`
- Modify: `tests/test_encoded_cache.py`

- [ ] **Step 1: Write failing lifecycle and capacity tests**

```python
# append to tests/test_encoded_cache.py
import json

from imgread_benchmark.encoded_cache import EncodedCacheError


def test_lru_evicts_idle_cache_but_keeps_active_lease(tmp_path):
    files = [
        _image(tmp_path / "one.jpg", b"1" * 4096),
        _image(tmp_path / "two.jpg", b"2" * 4096),
        _image(tmp_path / "three.jpg", b"3" * 4096),
    ]
    manager = _manager(tmp_path, limit=17 * 1024)
    first = manager.acquire([files[0]])
    with first:
        with manager.acquire([files[1]]):
            pass
        with manager.acquire([files[2]]) as third:
            assert third.hit is False
            assert len(third.evicted) == 1
        assert bytes(first.buffers[0]) == b"1" * 4096


def test_lower_limit_cannot_evict_active_cache(tmp_path):
    source = _image(tmp_path / "active.jpg", b"x" * 4096)
    other = _image(tmp_path / "other.jpg", b"y" * 4096)
    manager = _manager(tmp_path, limit=32 * 1024)
    active = manager.acquire([source])
    with active:
        smaller = _manager(tmp_path, limit=12 * 1024)
        with pytest.raises(EncodedCacheError, match="active caches"):
            smaller.acquire([other])


def test_corrupt_idle_cache_is_rebuilt(tmp_path):
    source = _image(tmp_path / "image.jpg", b"encoded")
    manager = _manager(tmp_path)
    with manager.acquire([source]) as first:
        cache_dir = first.index_path.parent
    value = json.loads((cache_dir / "index.json").read_text())
    value["data_bytes"] += 1
    (cache_dir / "index.json").write_text(json.dumps(value))

    with manager.acquire([source]) as rebuilt:
        assert rebuilt.hit is False
        assert bytes(rebuilt.buffers[0]) == b"encoded"


def test_interrupted_build_directory_is_cleaned(tmp_path):
    manager = _manager(tmp_path)
    incomplete = manager.options.root / ".build-dead"
    incomplete.mkdir(parents=True)
    source = _image(tmp_path / "image.jpg", b"encoded")
    with manager.acquire([source]):
        pass
    assert not incomplete.exists()


def test_tmpfs_check_rejects_regular_filesystem(tmp_path, monkeypatch):
    manager = EncodedCacheManager(
        CacheOptions(tmp_path / "cache", require_tmpfs=True)
    )
    monkeypatch.setattr(manager, "_filesystem_type", lambda _path: "ext4")
    source = _image(tmp_path / "image.jpg", b"encoded")
    with pytest.raises(EncodedCacheError, match="tmpfs"):
        manager.acquire([source])


def test_ram_reserve_blocks_build(tmp_path):
    source = _image(tmp_path / "image.jpg", b"x" * 1024)
    manager = EncodedCacheManager(
        CacheOptions(
            tmp_path / "cache",
            limit_bytes=2048,
            require_tmpfs=False,
            memory_probe=lambda: (1024, 512),
        )
    )
    with pytest.raises(EncodedCacheError, match="memory reserve"):
        manager.acquire([source])
```

- [ ] **Step 2: Run the lifecycle tests and verify failures describe missing policies**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_encoded_cache.py -q
```

Expected: the new eviction, corruption, tmpfs, and memory-reserve assertions fail.

- [ ] **Step 3: Implement mount detection, memory probing, nonblocking eviction, and corrupt-cache handling**

Add these rules to `EncodedCacheManager`:

```python
    @contextmanager
    def _exclusive_root_lock(self) -> Iterator[None]:
        lock = (self.options.root / ".root.lock").open("a+b")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _lock_path(self, cache_id: str) -> Path:
        return self.options.root / ".locks" / f"{cache_id}.lock"

    def _touch_usage(self, cache_id: str) -> None:
        marker = self.options.root / ".usage" / cache_id
        marker.touch()

    def _last_used(self, cache_id: str) -> int:
        marker = self.options.root / ".usage" / cache_id
        return marker.stat().st_mtime_ns if marker.exists() else 0

    def _try_remove(self, manifest: CacheManifest, cache_dir: Path) -> bool:
        lock = self._lock_path(manifest.cache_id).open("a+b")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            shutil.rmtree(cache_dir)
            (self.options.root / ".usage" / manifest.cache_id).unlink(missing_ok=True)
            return True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    @staticmethod
    def _filesystem_type(path: Path) -> str:
        resolved = path.resolve()
        best_mount = Path("/")
        best_type = ""
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mountpoint = Path(left.split()[4].replace("\\040", " "))
            try:
                resolved.relative_to(mountpoint)
            except ValueError:
                continue
            if len(mountpoint.parts) >= len(best_mount.parts):
                best_mount, best_type = mountpoint, right.split()[0]
        return best_type

    def _require_tmpfs(self, path: Path) -> None:
        fs_type = self._filesystem_type(path)
        if fs_type != "tmpfs":
            raise EncodedCacheError(f"decode-only cache must be on tmpfs; {path} is {fs_type}")

    def _probe_memory(self) -> tuple[int, int]:
        if self.options.memory_probe is not None:
            return self.options.memory_probe()
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) * 1024
        total, available = values["MemTotal"], values["MemAvailable"]
        maximum = Path("/sys/fs/cgroup/memory.max")
        current = Path("/sys/fs/cgroup/memory.current")
        if maximum.exists() and maximum.read_text().strip() != "max":
            cgroup_total = int(maximum.read_text())
            total = min(total, cgroup_total)
            available = min(available, cgroup_total - int(current.read_text()))
        return total, max(0, available)

```

Replace the temporary `_enforce_budget` implementation with the following
helpers. At the call site, pass a conservative build estimate instead of only
the payload length:

```python
    @staticmethod
    def _allocated_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(item.stat().st_blocks * 512 for item in path.rglob("*") if item.is_file())

    def _cache_usage(self) -> int:
        return sum(
            self._allocated_bytes(path)
            for pattern in ("cache-*", ".build-*")
            for path in self.options.root.glob(pattern)
        )

    def _free_tmpfs_bytes(self) -> int:
        stat = os.statvfs(self.options.root)
        return stat.f_bavail * stat.f_frsize

    def _memory_allows(self, requested_bytes: int) -> bool:
        total, available = self._probe_memory()
        reserve = max(512 * 1024**2, total // 10)
        return available - requested_bytes >= reserve

    def _valid_candidates(self) -> list[tuple[CacheManifest, Path]]:
        result = []
        for cache_dir in self.options.root.glob("cache-*"):
            try:
                manifest = CacheManifest.from_path(cache_dir / "index.json")
                self._validate_cache(cache_dir, manifest)
            except (EncodedCacheError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
            result.append((manifest, cache_dir))
        return result

    def _enforce_budget(
        self, requested_bytes: int, protected_ids: set[str]
    ) -> list[str]:
        if requested_bytes > self.options.limit_bytes:
            raise EncodedCacheError(
                f"cache requires {requested_bytes} bytes, limit is {self.options.limit_bytes}"
            )
        candidates = sorted(
            (
                (manifest, cache_dir)
                for manifest, cache_dir in self._valid_candidates()
                if manifest.cache_id not in protected_ids
            ),
            key=lambda item: (self._last_used(item[0].cache_id), item[0].cache_id),
        )
        evicted = []
        while (
            self._cache_usage() + requested_bytes > self.options.limit_bytes
            or self._free_tmpfs_bytes() < requested_bytes
            or not self._memory_allows(requested_bytes)
        ):
            if not candidates:
                total, available = self._probe_memory()
                reserve = max(512 * 1024**2, total // 10)
                if available - requested_bytes < reserve:
                    reason = "memory reserve"
                elif self._free_tmpfs_bytes() < requested_bytes:
                    reason = "tmpfs free space"
                else:
                    reason = "active caches"
                raise EncodedCacheError(f"cannot satisfy cache limit because of {reason}")
            manifest, cache_dir = candidates.pop(0)
            if self._try_remove(manifest, cache_dir):
                evicted.append(manifest.cache_id)
        return evicted

    @staticmethod
    def _build_estimate(entries: Sequence[SourceRecord]) -> int:
        page = 4096
        data = sum(entry.size for entry in entries)
        index_upper_bound = max(page, len(entries) * 512)
        return ((data + page - 1) // page) * page + index_upper_bound
```

Use `_build_estimate(normalized)` as `requested_bytes` for a miss. After writing
the actual index but before publication, verify that the finished allocated size
does not exceed the estimate; if it does, rerun capacity checks for the delta.
Filter superset candidates whose allocated directory is larger than the current
limit so a compact requested-only cache can be built instead.

When `_find_smallest_superset` catches an invalid candidate, derive its cache ID
from a valid `cache-<64 lowercase hex digits>` directory name and call a sibling
`_try_remove_path(cache_id, cache_dir)` helper. That helper uses the same
nonblocking exclusive lock and deletion body as `_try_remove`. If an invalid
candidate is active, leave it untouched. When its directory name is the exact
requested ID, raise `EncodedCacheError("requested cache is corrupt and active")`;
otherwise skip it and continue scanning.

- [ ] **Step 4: Run focused tests twice to expose leaked locks or mappings**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_encoded_cache.py -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_encoded_cache.py -q
```

Expected: both runs pass with no `BufferError`, lock hang, or leftover temporary directory.

- [ ] **Step 5: Commit cache lifecycle behavior**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check src/imgread_benchmark/encoded_cache.py tests/test_encoded_cache.py
git add src/imgread_benchmark/encoded_cache.py tests/test_encoded_cache.py
git commit -m "feat: manage decode cache lifecycle"
```

Expected: checks pass and the second cache commit is created.

---

### Task 3: Строгий ленивый реестр буферных декодеров

**Files:**
- Modify: `src/imgread_benchmark/read_img.py`
- Create: `tests/test_decode_registry.py`
- Modify: `tests/test_lazy_imports.py`

- [ ] **Step 1: Write failing registry tests**

```python
# tests/test_decode_registry.py
from types import SimpleNamespace

import pytest

from imgread_benchmark import read_img


@pytest.fixture(autouse=True)
def clear_decode_caches():
    for factory in (
        read_img.get_decode_img,
        read_img.get_decode_img_pil,
        read_img.get_decode_img_ndarray,
    ):
        factory.cache_clear()
    yield
    for factory in (
        read_img.get_decode_img,
        read_img.get_decode_img_pil,
        read_img.get_decode_img_ndarray,
    ):
        factory.cache_clear()


def test_decode_registry_includes_only_matching_functions(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        read_img,
        "get_img_libs",
        lambda: {
            "supported": SimpleNamespace(decode_img=lambda _data: expected),
            "path_only": SimpleNamespace(read_img=lambda _path: expected),
        },
    )
    assert read_img.get_decode_img()["supported"](memoryview(b"x")) is expected
    assert "path_only" not in read_img.get_decode_img()


def test_decode_registry_does_not_swallow_failures(monkeypatch):
    def fail(_data):
        raise ValueError("broken payload")

    monkeypatch.setattr(
        read_img,
        "get_img_libs",
        lambda: {"strict": SimpleNamespace(decode_img_ndarray=fail)},
    )
    with pytest.raises(ValueError, match="broken payload"):
        read_img.get_decode_img_ndarray()["strict"](memoryview(b"x"))
```

Append to `tests/test_lazy_imports.py`:

```python
def test_decode_mappings_are_lazy_mappings():
    import collections.abc
    from imgread_benchmark import read_img

    assert isinstance(read_img.decode_img, collections.abc.Mapping)
    assert isinstance(read_img.decode_img_pil, collections.abc.Mapping)
    assert isinstance(read_img.decode_img_ndarray, collections.abc.Mapping)
```

- [ ] **Step 2: Run tests and confirm missing registry names**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_registry.py tests/test_lazy_imports.py -q
```

Expected: failures mention missing `get_decode_img` and lazy mappings.

- [ ] **Step 3: Add strict registry factories without changing path readers**

Add to `src/imgread_benchmark/read_img.py`:

```python
def get_strict_func_dict(
    func_name: str, func_dict: dict[str, Any]
) -> dict[str, Callable[[memoryview], Any]]:
    return {
        lib_name: func
        for lib_name, adapter in func_dict.items()
        if (func := getattr(adapter, func_name, None)) is not None
    }


@lru_cache(maxsize=1)
def get_decode_img() -> Dict[str, Callable[[memoryview], Any]]:
    return get_strict_func_dict("decode_img", get_img_libs())


@lru_cache(maxsize=1)
def get_decode_img_pil() -> Dict[str, Callable[[memoryview], Image.Image]]:
    return get_strict_func_dict("decode_img_pil", get_img_libs())


@lru_cache(maxsize=1)
def get_decode_img_ndarray() -> Dict[str, Callable[[memoryview], ndarray]]:
    return get_strict_func_dict("decode_img_ndarray", get_img_libs())


decode_img = _LazyMapping(get_decode_img)
decode_img_pil = _LazyMapping(get_decode_img_pil)
decode_img_ndarray = _LazyMapping(get_decode_img_ndarray)
```

Keep `get_func_dict` and all `get_read_img*` factories unchanged so normal reads retain graceful degradation. Add decode factories to cache-clearing helpers in `tests/test_plugin_registry_integration.py` so plugin tests cannot leak registry state.

- [ ] **Step 4: Run registry and existing lazy/plugin tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_registry.py tests/test_lazy_imports.py tests/test_plugin_registry_integration.py tests/test_error_handling.py -q
```

Expected: all tests pass; ordinary path-reader error behavior is unchanged.

- [ ] **Step 5: Commit the registry**

Run:

```bash
git add src/imgread_benchmark/read_img.py tests/test_decode_registry.py tests/test_lazy_imports.py tests/test_plugin_registry_integration.py
git commit -m "feat: register strict buffer decoders"
```

Expected: one registry-focused commit.

---

### Task 4: Буферные адаптеры поддерживаемых библиотек

**Files:**
- Modify: `src/imgread_benchmark/img_libs/PIL.py`
- Modify: `src/imgread_benchmark/img_libs/cv2.py`
- Modify: `src/imgread_benchmark/img_libs/ajpegli.py`
- Modify: `src/imgread_benchmark/img_libs/imagecodecs.py`
- Modify: `src/imgread_benchmark/img_libs/jpeg4py.py`
- Modify: `src/imgread_benchmark/img_libs/simplejpeg.py`
- Modify: `src/imgread_benchmark/img_libs/turbojpeg.py`
- Modify: `src/imgread_benchmark/img_libs/imageio.py`
- Modify: `src/imgread_benchmark/img_libs/skimage.py`
- Modify: `src/imgread_benchmark/img_libs/imgread_rs.py`
- Modify: `src/imgread_benchmark/img_libs/local_rs.py`
- Create: `tests/test_decode_adapters.py`
- Modify: `tests/test_new_adapters.py`

- [ ] **Step 1: Write equivalence tests for every installed supported adapter**

```python
# tests/test_decode_adapters.py
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

SUPPORTED = (
    "PIL",
    "cv2",
    "ajpegli",
    "imagecodecs",
    "jpeg4py",
    "simplejpeg",
    "turbojpeg",
    "imageio",
    "skimage",
    "imgread_rs",
    "local_rs",
)
JPEG = Path(__file__).parent / "test_imgs" / "dog.jpg"


@pytest.mark.parametrize("name", SUPPORTED)
def test_buffer_ndarray_matches_same_adapter_file_read(name):
    try:
        adapter = import_module(f"imgread_benchmark.img_libs.{name}")
    except (ImportError, OSError):
        pytest.skip(f"{name} is unavailable")
    if not hasattr(adapter, "decode_img_ndarray"):
        pytest.fail(f"{name} has no decode_img_ndarray")
    payload = memoryview(JPEG.read_bytes())
    expected = np.asarray(adapter.read_img_ndarray(str(JPEG)))
    actual = np.asarray(adapter.decode_img_ndarray(payload))
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("name", SUPPORTED)
def test_buffer_pil_is_fully_loaded(name):
    try:
        adapter = import_module(f"imgread_benchmark.img_libs.{name}")
    except (ImportError, OSError):
        pytest.skip(f"{name} is unavailable")
    payload = bytearray(JPEG.read_bytes())
    image = adapter.decode_img_pil(memoryview(payload))
    payload[:] = b"\0" * len(payload)
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0
    image.getpixel((0, 0))
```

- [ ] **Step 2: Run the adapter tests and verify missing buffer functions**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_adapters.py -q
```

Expected: installed adapters fail with missing `decode_img_ndarray`.

- [ ] **Step 3: Implement Pillow, OpenCV, ajpegli, jpeg4py, simplejpeg, and turbojpeg buffer functions**

Use these exact native buffer calls, adding the three names to each module's `__all__`:

```python
# PIL.py
from io import BytesIO

def decode_img(data: memoryview) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        return image.convert("RGB")

decode_img_pil = decode_img

def decode_img_ndarray(data: memoryview) -> np.ndarray:
    return np.asarray(decode_img(data))


# cv2.py
def decode_img(data: memoryview) -> np.ndarray:
    encoded = np.frombuffer(data, dtype=np.uint8)
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR_RGB)
    if result is None:
        raise ValueError("OpenCV could not decode image buffer")
    return result

decode_img_ndarray = decode_img

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img(data))


# ajpegli.py
def decode_img(data: memoryview) -> np.ndarray:
    return ajpegli.decode(data, mode="RGB")

decode_img_ndarray = decode_img

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img(data))


# jpeg4py.py
def decode_img(data: memoryview) -> np.ndarray:
    return jpeg4py.JPEG(np.frombuffer(data, dtype=np.uint8)).decode()

decode_img_ndarray = decode_img

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img(data))


# simplejpeg.py
def decode_img(data: memoryview) -> np.ndarray:
    return simplejpeg.decode_jpeg(data, colorspace="RGB")

decode_img_ndarray = decode_img

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img(data))


# turbojpeg.py
def decode_img(data: memoryview) -> np.ndarray:
    return np.array(turbojpeg.decompress(data, pixelformat=turbojpeg.PF.RGB))

decode_img_ndarray = decode_img

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img(data))
```

The buffer-to-native conversion stays inside each function because it belongs inside the measured adapter path.

- [ ] **Step 4: Implement imagecodecs, imageio, skimage, imgread_rs, and local_rs functions**

```python
# imagecodecs.py
def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image

def read_img(img_path: str) -> np.ndarray:
    return _rgb(imagecodecs.imread(img_path))

def decode_img(data: memoryview) -> np.ndarray:
    return _rgb(imagecodecs.imread(bytes(data)))

decode_img_ndarray = decode_img

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img(data))


# imageio.py and skimage.py
from io import BytesIO

def decode_img(data: memoryview) -> np.ndarray:
    return imageio.v3.imread(BytesIO(data))  # use io.imread(BytesIO(data)) in skimage.py

decode_img_ndarray = decode_img

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img(data)).convert("RGB")


# imgread_rs.py
def decode_img_ndarray(data: memoryview) -> np.ndarray:
    result = _imgread_rs.load_numpy_from_bytes(data)
    if result is None:
        raise ValueError("imgread_rs returned no decoded image")
    return np.asarray(result)

decode_img = decode_img_ndarray

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img_ndarray(data), mode="RGB")


# local_rs.py
def decode_img_ndarray(data: memoryview) -> np.ndarray:
    try:
        result = _local_rs.open_jpeg_turbo_from_bytes(data)
    except Exception:
        result = _local_rs.open_jpeg_from_bytes(data)
    if result is None:
        raise ValueError("local_rs returned no decoded image")
    return np.asarray(result)

decode_img = decode_img_ndarray

def decode_img_pil(data: memoryview) -> Image.Image:
    return Image.fromarray(decode_img_ndarray(data), mode="RGB")
```

For `imageio.py`, keep its existing module import and call `imageio.v3.imread`.
For `skimage.py`, its existing alias `import skimage.io as io` means the call is
`io.imread`. The snippets already check the native APIs in this set that can
return no image (`cv2`, `imgread_rs`, and `local_rs`). Preserve each existing
`read_img*` implementation except the shared `_rgb` extraction shown for
imagecodecs.

- [ ] **Step 5: Run adapter, normal-reader, and optional-dependency tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_adapters.py tests/test_new_adapters.py tests/test_image_libs.py tests/test_refactored_libs.py -q
```

Expected: installed libraries pass buffer/file equivalence; unavailable optional libraries skip using existing runtime checks.

- [ ] **Step 6: Commit buffer adapters**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check src/imgread_benchmark/img_libs tests/test_decode_adapters.py
git add src/imgread_benchmark/img_libs tests/test_decode_adapters.py tests/test_new_adapters.py
git commit -m "feat: decode images from memory buffers"
```

Expected: Ruff and adapter tests pass; all eleven adapter changes are in one interface-focused commit.

---

### Task 5: Последовательный исполнитель с корректной границей таймера

**Files:**
- Create: `src/imgread_benchmark/decode_benchmark.py`
- Create: `tests/test_decode_benchmark.py`

- [ ] **Step 1: Write tests for warmup, timing, strict errors, selection, and result accounting**

```python
# tests/test_decode_benchmark.py
import pytest

from imgread_benchmark.decode_benchmark import DecodeBenchmark


def test_warmup_is_outside_timed_repeats():
    calls = []

    def decode(data):
        calls.append(bytes(data))
        return object()

    benchmark = DecodeBenchmark(
        buffers=(memoryview(b"a"), memoryview(b"b")),
        func_dict={"decoder": decode},
        num_repeats=3,
        clear_progress=True,
    )
    benchmark.run()
    assert calls == [b"a", b"b"] * 4  # one warmup plus three timed passes
    assert len(benchmark._results["decoder"]) == 3
    assert benchmark.results["decoder"] >= 0


def test_failed_warmup_has_no_numeric_result():
    def fail(data):
        if bytes(data) == b"bad":
            raise ValueError("invalid image")
        return object()

    benchmark = DecodeBenchmark(
        buffers=(memoryview(b"ok"), memoryview(b"bad")),
        func_dict={"broken": fail, "good": lambda _data: object()},
        num_repeats=2,
        clear_progress=True,
    )
    benchmark.run()
    assert "broken" not in benchmark.results
    assert benchmark.exceptions["broken"][0]["index"] == 1
    assert "good" in benchmark.results


def test_none_is_a_decode_failure():
    benchmark = DecodeBenchmark(
        buffers=(memoryview(b"x"),),
        func_dict={"decoder": lambda _data: None},
        num_repeats=1,
        clear_progress=True,
    )
    benchmark.run()
    assert "decoder" not in benchmark.results
    assert "returned None" in benchmark.exceptions["decoder"][0]["error"]


def test_func_name_and_exclude_follow_existing_contract():
    benchmark = DecodeBenchmark(
        buffers=(memoryview(b"x"),),
        func_dict={"one": lambda _data: object(), "two": lambda _data: object()},
        num_repeats=1,
        clear_progress=True,
    )
    benchmark.run(func_name="one")
    assert set(benchmark.results) == {"one"}
    benchmark.run(exclude="one")
    assert set(benchmark.results) == {"two"}
```

- [ ] **Step 2: Run the tests and verify the executor is missing**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_benchmark.py -q
```

Expected: collection fails for missing `decode_benchmark`.

- [ ] **Step 3: Implement the sequential executor**

```python
# src/imgread_benchmark/decode_benchmark.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

from rich import print as rprint
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn


class DecodeBenchmark:
    def __init__(
        self,
        *,
        buffers: Sequence[memoryview],
        func_dict: dict[str, Callable[[memoryview], Any]],
        num_repeats: int = 5,
        clear_progress: bool = False,
    ) -> None:
        self.buffers = tuple(buffers)
        self.func_dict = func_dict
        self.num_repeats = num_repeats
        self.clear_progress = clear_progress
        self._results: dict[str, list[float]] = {}
        self.exceptions: dict[str, list[dict[str, object]]] = defaultdict(list)

    @property
    def num_items(self) -> int:
        return len(self.buffers)

    @property
    def results(self) -> dict[str, float]:
        return {
            name: sum(samples) / len(samples)
            for name, samples in self._results.items()
            if samples
        }

    def _selected(self, func_name: str | None, exclude: str | None) -> list[str]:
        if func_name is not None:
            return [func_name] if func_name in self.func_dict else []
        return [name for name in self.func_dict if name != exclude]

    @staticmethod
    def _decode_one(func, data: memoryview, index: int):
        result = func(data)
        if result is None:
            raise ValueError(f"decoder returned None for image {index}")
        return result

    def _pass(self, name: str) -> None:
        func = self.func_dict[name]
        for index, data in enumerate(self.buffers):
            try:
                result = self._decode_one(func, data, index)
                del result
            except Exception as exc:
                raise DecodePassError(index, exc) from exc

    def _record_failure(self, name: str, index: int, exc: Exception) -> None:
        self.exceptions[name].append(
            {"index": index, "error": f"{type(exc).__name__}: {exc}"}
        )

    def _warm(self, name: str) -> bool:
        func = self.func_dict[name]
        for index, data in enumerate(self.buffers):
            try:
                result = self._decode_one(func, data, index)
                del result
            except Exception as exc:
                self._record_failure(name, index, exc)
                return False
        return True

    def run(
        self,
        func_name: str | None = None,
        exclude: str | None = None,
        **_compat: object,
    ) -> None:
        self._results = {}
        self.exceptions = defaultdict(list)
        names = self._selected(func_name, exclude)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            transient=self.clear_progress,
        ) as progress:
            for name in names:
                if not self._warm(name):
                    continue
                task = progress.add_task(name, total=self.num_repeats)
                samples: list[float] = []
                for _repeat in range(self.num_repeats):
                    started = perf_counter()
                    try:
                        self._pass(name)
                    except DecodePassError as exc:
                        self._record_failure(name, exc.index, exc.cause)
                        samples = []
                        break
                    samples.append(perf_counter() - started)
                    progress.update(task, advance=1)
                if samples:
                    self._results[name] = samples
        self.print_results_per_item()

    def print_results_per_item(self) -> None:
        if not self._results:
            rprint("No successful decode-only results.")
            return
        rprint(" Func name  | Decoded images/sec")
        speeds = {
            name: self.num_items / average for name, average in self.results.items()
        }
        for name in sorted(speeds, key=speeds.get, reverse=True):
            rprint(f"{name:12}: {speeds[name]:.2f}")
```

Define the exception used by `_pass` immediately above `DecodeBenchmark`:

```python
class DecodePassError(RuntimeError):
    def __init__(self, index: int, cause: Exception) -> None:
        super().__init__(str(cause))
        self.index = index
        self.cause = cause
```

In `_selected`, print the same `<name> is not in func_dict` message as
`Benchmark` for a missing explicit name. In `run`, print `Nothing to test` and
return before creating `Progress` when `_selected` is empty. The progress update
remains after `perf_counter()` is read.

- [ ] **Step 4: Run sequential executor tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_benchmark.py -q
```

Expected: all tests pass and timing result count equals `num_repeats`.

- [ ] **Step 5: Commit the executor**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check src/imgread_benchmark/decode_benchmark.py tests/test_decode_benchmark.py
git add src/imgread_benchmark/decode_benchmark.py tests/test_decode_benchmark.py
git commit -m "feat: benchmark sequential buffer decoding"
```

Expected: focused tests and Ruff pass.

---

### Task 6: Multiprocessing с общим mapping и готовым пулом

**Files:**
- Modify: `src/imgread_benchmark/decode_benchmark.py`
- Create: `tests/test_decode_multiprocessing.py`

- [ ] **Step 1: Write a real spawn test using the shared mapping**

```python
# tests/test_decode_multiprocessing.py
from pathlib import Path

from PIL import Image

from imgread_benchmark.decode_benchmark import MultiprocessingDecodeBenchmark
from imgread_benchmark.encoded_cache import CacheOptions, EncodedCacheManager


def _jpeg(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (8, 8), color).save(path, format="JPEG")
    return path


def test_spawn_workers_open_shared_mapping_and_reuse_pool(tmp_path):
    files = [_jpeg(tmp_path / f"{index}.jpg", (index, 2, 3)) for index in range(3)]
    manager = EncodedCacheManager(
        CacheOptions(
            tmp_path / "cache",
            limit_bytes=1024 * 1024,
            require_tmpfs=False,
            memory_probe=lambda: (1 << 40, 1 << 40),
        )
    )
    with manager.acquire(files) as lease:
        benchmark = MultiprocessingDecodeBenchmark(
            data_path=lease.data_path,
            ranges=lease.ranges,
            decoder_names=("PIL",),
            target_format="np",
            num_repeats=2,
            num_workers=2,
            clear_progress=True,
            lock_path=lease.lock_path,
        )
        benchmark.run()
    assert len(benchmark._results["PIL"]) == 2
    assert not benchmark.exceptions
```

Expose read-only `ranges` and `lock_path` properties on `EncodedCacheLease`; `ranges` is the ordered tuple of `(offset, length)` pairs and contains no memoryview.

```python
# add to EncodedCacheLease
    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        return self._ranges

    @property
    def lock_path(self) -> Path:
        return Path(self._lock_file.name)
```

- [ ] **Step 2: Run the multiprocessing test and verify the class is missing**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_multiprocessing.py -q
```

Expected: import fails for `MultiprocessingDecodeBenchmark`.

- [ ] **Step 3: Add module-level spawn worker state and functions**

```python
# append to src/imgread_benchmark/decode_benchmark.py
import fcntl
import mmap
import multiprocessing
from multiprocessing.queues import Queue
import os
from pathlib import Path
import queue
import time

_WORKER_MAPPING: mmap.mmap | None = None
_WORKER_FILE = None
_WORKER_LOCK = None
_WORKER_RANGES: tuple[tuple[int, int], ...] = ()
_WORKER_DECODER = None


def _decoder_factory(target_format: str):
    from .read_img import get_decode_img, get_decode_img_ndarray, get_decode_img_pil

    return {
        "def": get_decode_img,
        "np": get_decode_img_ndarray,
        "pil": get_decode_img_pil,
    }[target_format]


def _worker_init(
    data_path: str,
    ranges: tuple[tuple[int, int], ...],
    lock_path: str,
    decoder_name: str,
    target_format: str,
    ready: Queue,
) -> None:
    global _WORKER_MAPPING, _WORKER_FILE, _WORKER_LOCK, _WORKER_RANGES, _WORKER_DECODER
    try:
        _WORKER_LOCK = open(lock_path, "a+b")
        fcntl.flock(_WORKER_LOCK, fcntl.LOCK_SH)
        _WORKER_FILE = open(data_path, "rb")
        _WORKER_MAPPING = mmap.mmap(_WORKER_FILE.fileno(), 0, access=mmap.ACCESS_READ)
        _WORKER_RANGES = ranges
        _WORKER_DECODER = _decoder_factory(target_format)()[decoder_name]
        for offset in range(0, len(_WORKER_MAPPING), mmap.PAGESIZE):
            _WORKER_MAPPING[offset]
        probe = _worker_decode(0)
        if probe is not None:
            raise RuntimeError(probe)
        ready.put(("ready", os.getpid(), ""))
    except Exception as exc:
        ready.put(("error", os.getpid(), f"{type(exc).__name__}: {exc}"))
        raise


def _worker_decode(index: int) -> str | None:
    try:
        offset, length = _WORKER_RANGES[index]
        data = memoryview(_WORKER_MAPPING)[offset : offset + length]
        try:
            result = _WORKER_DECODER(data)
            if result is None:
                return f"decoder returned None for image {index}"
            del result
            return None
        finally:
            data.release()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
```

If the selection is empty, reject it before creating workers so `_worker_decode(0)` is never called.

- [ ] **Step 4: Implement pool lifecycle, readiness, warmup, and timed passes**

Add `MultiprocessingDecodeBenchmark` with this public constructor and algorithm:

```python
class MultiprocessingDecodeBenchmark(DecodeBenchmark):
    READY_TIMEOUT_SECONDS = 120

    def __init__(
        self,
        *,
        data_path: Path,
        ranges: Sequence[tuple[int, int]],
        lock_path: Path,
        decoder_names: Sequence[str],
        target_format: str,
        num_repeats: int,
        num_workers: int | None,
        clear_progress: bool,
    ) -> None:
        super().__init__(
            buffers=(),
            func_dict={name: lambda _data: None for name in decoder_names},
            num_repeats=num_repeats,
            clear_progress=clear_progress,
        )
        self.data_path = data_path
        self.ranges = tuple(ranges)
        self.lock_path = lock_path
        self.decoder_names = tuple(decoder_names)
        self.target_format = target_format
        cpu_count = os.cpu_count() or 1
        self.num_workers = cpu_count if num_workers is None else min(num_workers, cpu_count)

    @property
    def num_items(self) -> int:
        return len(self.ranges)

    def _open_pool(self, name: str):
        context = multiprocessing.get_context("spawn")
        ready = context.Queue()
        pool = context.Pool(
            self.num_workers,
            initializer=_worker_init,
            initargs=(
                str(self.data_path),
                self.ranges,
                str(self.lock_path),
                name,
                self.target_format,
                ready,
            ),
        )
        deadline = time.monotonic() + self.READY_TIMEOUT_SECONDS
        pids = set()
        while len(pids) < self.num_workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pool.terminate()
                pool.join()
                raise RuntimeError(f"workers did not become ready for {name}")
            try:
                status, pid, detail = ready.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if any(process.exitcode not in (None, 0) for process in pool._pool):
                    pool.terminate()
                    pool.join()
                    raise RuntimeError(f"worker exited while initializing {name}")
                continue
            if status == "error":
                pool.terminate()
                pool.join()
                raise RuntimeError(f"worker {pid} could not initialize {name}: {detail}")
            pids.add(pid)
        return pool
```

Do not expose private `pool._pool` use outside a small `_worker_died(pool)` helper marked `# pragma: no cover` for platform-specific process failure. `_run_pool_pass` calls `pool.map(_worker_decode, range(len(ranges)), chunksize=max(1, len(ranges) // (num_workers * 8)))`, collects every non-`None` error with its input index, and returns success. Call it once before timing. For each measured repeat, take `perf_counter()` immediately before `pool.map` and immediately after all statuses arrive; update Rich progress afterwards. Reuse the pool for all repeats of one backend, then `close()`/`join()` in success and `terminate()`/`join()` on failure. Build a new pool for the next backend.

Custom `func_dict` is supported only by the sequential executor; the multiprocessing class resolves installed built-in or entry-point decoders by registry name in each spawned process.

- [ ] **Step 5: Run spawn and sequential regression tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_multiprocessing.py tests/test_decode_benchmark.py tests/test_encoded_cache.py -q
```

Expected: tests pass without hang; pytest exits normally and removes temporary directories.

- [ ] **Step 6: Commit multiprocessing support**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check src/imgread_benchmark/decode_benchmark.py src/imgread_benchmark/encoded_cache.py tests/test_decode_multiprocessing.py
git add src/imgread_benchmark/decode_benchmark.py src/imgread_benchmark/encoded_cache.py tests/test_decode_multiprocessing.py
git commit -m "feat: benchmark buffers with worker pool"
```

Expected: one commit with shared-mapping multiprocessing support.

---

### Task 7: Интеграция `BenchmarkImgRead` и CLI

**Files:**
- Modify: `src/imgread_benchmark/benchmark.py`
- Modify: `src/imgread_benchmark/cli.py`
- Modify: `tests/test_benchmark_lazy.py`
- Modify: `tests/test_cli_unified.py`

- [ ] **Step 1: Write failing API tests proving lazy preparation and result propagation**

```python
# append to tests/test_benchmark_lazy.py
def test_decode_only_constructor_does_not_build_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "imgread_benchmark.encoded_cache.EncodedCacheManager.acquire",
        lambda *_args, **_kwargs: calls.append("acquire"),
    )
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=[str(tmp_path / "image.jpg")],
        target_format="np",
        func_dict={"dummy": lambda _data: object()},
        decode_only=True,
        cache_dir=tmp_path / "cache",
        cache_limit=1024,
    )
    assert calls == []
    assert bench.decode_only is True


def test_decode_only_run_copies_runner_results(tmp_path, monkeypatch):
    source = tmp_path / "image.jpg"
    source.write_bytes(b"encoded")

    class DummyRunner:
        def __init__(self, **_kwargs):
            self._results = {}
            self.exceptions = {}
        @property
        def results(self):
            return {"dummy": 1.25}
        def run(self, **_kwargs):
            self._results = {"dummy": [1.0, 1.5]}

    monkeypatch.setattr("imgread_benchmark.benchmark.DecodeBenchmark", DummyRunner)
    bench = benchmark_mod.BenchmarkImgRead(
        filenames=[str(source)],
        target_format="np",
        func_dict={"dummy": lambda _data: object()},
        num_repeats=2,
        decode_only=True,
        cache_dir=tmp_path / "cache",
        cache_limit=1024 * 1024,
        require_tmpfs=False,
    )
    bench.run()
    assert bench._results == {"dummy": [1.0, 1.5]}
```

`require_tmpfs` is a keyword-only internal/test parameter on `BenchmarkImgRead`, defaulting to `True`; it is not exposed as a CLI flag.

- [ ] **Step 2: Write CLI parsing and delegation tests**

```python
# append to tests/test_cli_unified.py
def test_normalize_decode_only_flags_before_path():
    argv = ["--decode-only", "--cache-limit", "1GiB", "/imgs"]
    assert _normalize_argv(argv) == ["benchmark", *argv]


def test_cache_options_require_decode_only(tmp_path, capsys):
    cli = _build_cli()
    with pytest.raises(SystemExit) as exc:
        cli(["benchmark", str(tmp_path), "--cache-limit", "1GiB"])
    assert exc.value.code == 2
    assert "require --decode-only" in capsys.readouterr().err


def test_decode_only_options_reach_benchmark(tmp_path, monkeypatch):
    (tmp_path / "image.jpg").write_bytes(b"encoded")
    seen = {}

    class DummyBench:
        func_dict = {"dummy": int}
        exceptions = {}
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def run(self, **kwargs):
            seen["run"] = kwargs

    monkeypatch.setattr("imgread_benchmark.benchmark.BenchmarkImgRead", DummyBench)
    _build_cli()([
        "benchmark", str(tmp_path), "--decode-only",
        "--cache-dir", str(tmp_path / "cache"), "--cache-limit", "1MiB",
    ])
    assert seen["decode_only"] is True
    assert seen["cache_limit"] == 1024**2
    assert seen["cache_dir"] == tmp_path / "cache"
```

- [ ] **Step 3: Run focused tests and confirm missing parameters and flags**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_benchmark_lazy.py tests/test_cli_unified.py -q
```

Expected: new assertions fail on unknown constructor keywords or unknown CLI options.

- [ ] **Step 4: Integrate the decode runner into `BenchmarkImgRead`**

At module scope in `benchmark.py`, import `Path`, `EncodedCacheManager`, `CacheOptions`, `DecodeBenchmark`, and `MultiprocessingDecodeBenchmark`. Add:

```python
def _get_decode_to_format() -> Dict[str, Callable]:
    from .read_img import get_decode_img, get_decode_img_ndarray, get_decode_img_pil

    return {
        "def": get_decode_img(),
        "pil": get_decode_img_pil(),
        "np": get_decode_img_ndarray(),
    }
```

Extend `BenchmarkImgRead.__init__` after existing arguments:

```python
        *,
        decode_only: bool = False,
        cache_dir: str | Path | None = None,
        cache_limit: int = 2 * 1024**3,
        require_tmpfs: bool = True,
```

Store the complete filename list, decode settings, and whether `func_dict` was supplied. In decode mode, choose `_get_decode_to_format()[target_format]` unless `func_dict` was supplied, but do not construct `EncodedCacheManager` or touch files in `__init__`. Keep the current `super().__init__` call to preserve attributes and normal behavior.

Override `run` with this dispatch shape:

```python
    def run(self, func_name=None, exclude=None, num_repeats=None,
            num_samples=None, multiprocessing=None, num_workers=None):
        if not self.decode_only:
            return super().run(
                func_name=func_name, exclude=exclude, num_repeats=num_repeats,
                num_samples=num_samples, multiprocessing=multiprocessing,
                num_workers=num_workers,
            )
        repeats = self.num_repeats if num_repeats is None else num_repeats
        filenames = self.item_list if num_samples is None else self.item_list[:num_samples]
        manager = EncodedCacheManager(
            CacheOptions(
                root=self.cache_dir,
                limit_bytes=self.cache_limit,
                require_tmpfs=self.require_tmpfs,
            )
        )
        with manager.acquire(filenames) as lease:
            lease.prefault()
            self.cache_info = {
                "cache_id": lease.cache_id,
                "hit": lease.hit,
                "data_bytes": lease.data_bytes,
                "preparation_seconds": lease.preparation_seconds,
                "evicted": lease.evicted,
                "path": str(lease.data_path.parent),
            }
            if multiprocessing:
                if self._custom_decode_funcs:
                    raise ValueError("custom decode func_dict is supported only without multiprocessing")
                runner = MultiprocessingDecodeBenchmark(
                    data_path=lease.data_path,
                    ranges=lease.ranges,
                    lock_path=lease.lock_path,
                    decoder_names=tuple(self.func_dict),
                    target_format=self.target_format,
                    num_repeats=repeats,
                    num_workers=num_workers,
                    clear_progress=self.clear_progress,
                )
            else:
                runner = DecodeBenchmark(
                    buffers=lease.buffers,
                    func_dict=self.func_dict,
                    num_repeats=repeats,
                    clear_progress=self.clear_progress,
                )
            runner.run(func_name=func_name, exclude=exclude)
            self._results = runner._results
            self.exceptions = runner.exceptions
```

Resolve default `cache_dir` to `Path("/dev/shm") / f"imgread_benchmark-{os.getuid()}"`. Initialize `cache_info` to `None`. Include `decode_only` in `func_names`. In `target_format.setter`, choose the read or decode registry according to mode. Before acquiring the cache, select requested functions and fail if an explicitly requested decoder is absent or if nothing is supported, so unsupported requests do not copy data.

- [ ] **Step 5: Add CLI fields, allowlists, validation, and error reporting**

Add `--decode-only` to `_BENCHMARK_FLAG_ALLOWLIST`; add `--cache-dir` and `--cache-limit` to that set and `_FLAGS_WITH_VALUES`. Add dataclass fields:

```python
        decode_only: bool = field_argument(
            "--decode-only", default=False, action="store_true",
            help="benchmark decoding from a prepared tmpfs memmap cache",
        )
        cache_dir: str | None = field_argument(
            "--cache-dir", default=None,
            help="tmpfs directory for the decode-only cache",
        )
        cache_limit: str | None = field_argument(
            "--cache-limit", default=None,
            help="total decode cache limit, default 2GiB",
        )
```

In `benchmark(cfg)`, reject cache options when `decode_only` is false. Parse an
explicit value with `parse_size`; convert `ValueError` into `SystemExit(2)` and
an actionable stderr message. Pass decoded settings into `BenchmarkImgRead`.
Skip `_get_multiprocessing_compat_errors` in decode-only mode because spawned
workers resolve decoders by name. Catch `EncodedCacheError` and decode runner
setup errors, print `Error: decode-only benchmark could not start: ...`, and
exit 2.

Immediately after acquiring and prefaulting the lease, `BenchmarkImgRead.run`
prints a compact cache preparation line from `self.cache_info`, before calling
either decode runner. This keeps preparation output outside the timer and ahead
of benchmark results. The CLI does not print the line a second time. After the
run, CLI returns exit 1 if `bench.exceptions` is nonempty, after successful
results and an error summary have already been printed.

Before acquiring the cache, compare the full read registry with the chosen
decode registry. In all-libraries mode, print each unavailable combination as
`Skipping <name>: no <target_format> buffer decoder`. For explicit `-l`, raise
an actionable error if that name is absent. If filtering leaves no decoders,
raise before any source file is copied.

- [ ] **Step 6: Run API and CLI tests, then the complete CLI regression file**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_benchmark_lazy.py tests/test_cli_unified.py -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_cli_unified.py -q
```

Expected: all tests pass; existing multiprocessing error messages still pass in file mode.

- [ ] **Step 7: Commit API and CLI integration**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check src/imgread_benchmark/benchmark.py src/imgread_benchmark/cli.py tests/test_benchmark_lazy.py tests/test_cli_unified.py
git add src/imgread_benchmark/benchmark.py src/imgread_benchmark/cli.py tests/test_benchmark_lazy.py tests/test_cli_unified.py
git commit -m "feat: expose decode-only benchmark mode"
```

Expected: one integration commit; `src/imgread_benchmark/cl_app.py` remains unchanged because it is not the installed unified CLI entry point.

---

### Task 8: End-to-end behavior, documentation, and final verification

**Files:**
- Modify: `README.md`
- Create: `tests/test_decode_integration.py`

- [ ] **Step 1: Write end-to-end sequential cache-hit and CLI failure tests**

```python
# tests/test_decode_integration.py
from pathlib import Path

from PIL import Image
import pytest

from imgread_benchmark.benchmark import BenchmarkImgRead


def _dataset(path: Path, count: int = 3) -> list[str]:
    path.mkdir()
    filenames = []
    for index in range(count):
        target = path / f"{index}.jpg"
        Image.new("RGB", (16, 16), (index, 20, 30)).save(target, format="JPEG")
        filenames.append(str(target))
    return filenames


def test_second_decode_only_run_hits_cache(tmp_path):
    files = _dataset(tmp_path / "images")
    options = dict(
        filenames=files,
        target_format="np",
        num_repeats=1,
        decode_only=True,
        cache_dir=tmp_path / "cache",
        cache_limit=1024 * 1024,
        require_tmpfs=False,
        clear_progress=True,
    )
    first = BenchmarkImgRead(**options)
    first.run(func_name="PIL")
    second = BenchmarkImgRead(**options)
    second.run(func_name="PIL")
    assert first.cache_info["hit"] is False
    assert second.cache_info["hit"] is True
    assert set(second.results) == {"PIL"}


def test_smaller_run_reuses_larger_cache(tmp_path):
    files = _dataset(tmp_path / "images")
    options = dict(
        target_format="np", num_repeats=1, decode_only=True,
        cache_dir=tmp_path / "cache", cache_limit=1024 * 1024,
        require_tmpfs=False, clear_progress=True,
    )
    large = BenchmarkImgRead(filenames=files, **options)
    large.run(func_name="PIL")
    small = BenchmarkImgRead(filenames=files[:1], **options)
    small.run(func_name="PIL")
    assert small.cache_info["hit"] is True
    assert small.cache_info["cache_id"] == large.cache_info["cache_id"]
```

- [ ] **Step 2: Run the end-to-end tests**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_decode_integration.py -q
```

Expected: both tests pass and the second run reports a hit.

- [ ] **Step 3: Document the user workflow and measurement semantics**

Add this section to `README.md` after Benchmarking:

```markdown
## Decode-only benchmarking

On Linux, `--decode-only` copies the selected compressed image files into a
reusable read-only memory-mapped cache on tmpfs before timing. Cache preparation,
opening the mapping, page warmup, library imports, progress rendering, and worker
startup are excluded from the measured time. Required buffer conversion inside
an adapter and multiprocessing task coordination are included.

```bash
uv run imgread_benchmark /path/to/images --decode-only -n 3000 -t np
uv run imgread_benchmark /path/to/images --decode-only -n 200 -t np
```

The second command can reuse the larger cache. The default cache is
`/dev/shm/imgread_benchmark-<uid>` with a total `2GiB` limit. Override it with:

```bash
uv run imgread_benchmark /path/to/images --decode-only \
  --cache-dir /path/on/tmpfs --cache-limit 4GiB
```

Completed caches survive process exit and are removed by LRU when space is
needed. They disappear when the tmpfs is cleared or the machine reboots.
Supported buffer decoders are listed by the command before a run; explicitly
selecting a path-only backend is an error. A custom cache directory must be on
tmpfs. This mode does not pin pages with `mlock`, so swap or later memory pressure
can still affect the host.
```

Also add `--decode-only`, `--cache-dir`, and `--cache-limit` to the existing options list.

- [ ] **Step 4: Run all tests and lint once**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check .
```

Expected: pytest completes with all required tests passing and only established optional-dependency skips; Ruff reports no errors.

- [ ] **Step 5: Run a real smoke test on tmpfs and confirm reuse**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run imgread_benchmark tests/test_imgs --decode-only -n 2 -t np -l PIL -r 1
UV_CACHE_DIR=.uv-cache uv run imgread_benchmark tests/test_imgs --decode-only -n 1 -t np -l PIL -r 1
```

Expected: both commands succeed, print decoded images/sec, and the second output reports a cache hit using the first command's cache ID. Cache preparation time is printed separately.

- [ ] **Step 6: Inspect the final diff and commit documentation/integration coverage**

Run:

```bash
git diff --check
git status --short
git add README.md tests/test_decode_integration.py
git commit -m "docs: explain decode-only benchmarking"
```

Expected: no whitespace errors; the final commit contains README and end-to-end coverage. The pre-existing untracked `scripts/compare_imgread_versions.py` remains untouched and uncommitted.

- [ ] **Step 7: Record final evidence for review**

Run:

```bash
git log --oneline -8
git status --short
```

Expected: the decode-only work appears as the planned sequence of focused commits; the working tree contains no new uncommitted files from this feature.
