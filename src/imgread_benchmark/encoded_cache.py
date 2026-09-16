"""Immutable encoded-file caches on tmpfs, with process-owned read leases."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_CACHE_LIMIT = 2 * 1024**3
_CACHE_ID = re.compile(r"[0-9a-f]{64}\Z")


class EncodedCacheError(RuntimeError):
    pass


def parse_cache_limit(value):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"([0-9]+)(KiB|MiB|GiB)?", value)
        if match:
            size = (
                int(match[1])
                * {None: 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[match[2]]
            )
            if size > 0:
                return size
    raise ValueError("cache limit must be positive bytes or a size such as 512MiB")


def _mounts():
    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)

    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields, extra = before.split(), after.split()
        yield Path(unescape(fields[4])), unescape(fields[3]), extra[0], extra[2]


def _require_tmpfs(root):
    matches = [
        (path, kind) for path, _, kind, _ in _mounts() if root.is_relative_to(path)
    ]
    if not matches or max(matches, key=lambda item: len(str(item[0])))[1] != "tmpfs":
        raise EncodedCacheError(f"decode-only cache must be on tmpfs: {root}")


def _memory_budget():
    values = {
        line.split(":", 1)[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemTotal:", "MemAvailable:"))
    }
    total, available = values["MemTotal"], values["MemAvailable"]
    groups = [
        line.split(":", 2)
        for line in Path("/proc/self/cgroup").read_text().splitlines()
    ]
    for mount, mount_root, kind, options in _mounts():
        if kind not in ("cgroup", "cgroup2") or (
            kind == "cgroup" and "memory" not in options.split(",")
        ):
            continue
        group = next(
            (
                path
                for _, controllers, path in groups
                if (kind == "cgroup2" and not controllers)
                or "memory" in controllers.split(",")
            ),
            None,
        )
        if group is None or not Path(group).is_relative_to(mount_root):
            continue
        current = mount / Path(group).relative_to(mount_root)
        while current.is_relative_to(mount):
            limit_name, used_name = (
                ("memory.max", "memory.current")
                if kind == "cgroup2"
                else ("memory.limit_in_bytes", "memory.usage_in_bytes")
            )
            try:
                raw = (current / limit_name).read_text().strip()
                if raw != "max":
                    limit = int(raw)
                    if limit < 2**60:
                        used = int((current / used_name).read_text())
                        total = min(total, limit)
                        available = min(available, max(0, limit - used))
            except FileNotFoundError:
                pass
            if current == mount:
                break
            current = current.parent
    return max(0, available - max(512 * 1024**2, total // 10))


def _identity(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise EncodedCacheError(f"source must be a nonempty regular file: {path}")
    return dict(size=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def _key(records):
    raw = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _lock(path, *, exclusive, blocking=True):
    import fcntl

    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stream = os.fdopen(descriptor, "r+b")
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(stream, operation | (0 if blocking else fcntl.LOCK_NB))
    except BaseException:
        stream.close()
        raise
    return stream


@contextmanager
def _root_lock(root):
    until = time.monotonic() + 30
    while True:
        try:
            lock = _lock(root / ".root.lock", exclusive=True, blocking=False)
            break
        except BlockingIOError:
            if time.monotonic() >= until:
                raise EncodedCacheError(
                    f"cache preparation lock is busy: {root}"
                ) from None
            time.sleep(0.025)
    try:
        yield
    finally:
        lock.close()


def _allocated(path):
    if path.is_symlink():
        raise EncodedCacheError(f"unexpected symlink in cache: {path}")
    if path.is_dir():
        return sum(_allocated(item) for item in path.iterdir())
    return path.stat().st_blocks * 512


def _datasets(root):
    return [path for path in root.iterdir() if _CACHE_ID.fullmatch(path.name)]


def _usage(root):
    return sum(
        _allocated(path)
        for path in root.iterdir()
        if _CACHE_ID.fullmatch(path.name) or path.name.startswith(".build-")
    )


def _discard(root, path):
    try:
        lease = _lock(root / ".locks" / path.name, exclusive=True, blocking=False)
    except BlockingIOError:
        return False
    try:
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
        (root / ".used" / path.name).unlink(missing_ok=True)
        # Lock files are retained: deleting them could split leases across inodes.
        return True
    finally:
        lease.close()


def _make_room(root, needed, limit, protected=()):
    evicted = []

    def last_use(path):
        try:
            return (root / ".used" / path.name).stat().st_mtime_ns
        except FileNotFoundError:
            return 0

    for path in sorted(_datasets(root), key=last_use):
        if _usage(root) + needed <= limit:
            break
        if path.name not in protected and _discard(root, path):
            evicted.append(path.name)
    if _usage(root) + needed > limit:
        raise EncodedCacheError(
            f"cache needs {needed} additional bytes; limit={limit}, used={_usage(root)}; active caches cannot be evicted"
        )
    return evicted


def _read_index(path):
    try:
        if (
            path.is_symlink()
            or (path / "index.json").is_symlink()
            or (path / "data.bin").is_symlink()
        ):
            raise ValueError("symlink")
        index = json.loads((path / "index.json").read_text())
        if index["schema_version"] != 1 or index["cache_id"] != path.name:
            raise ValueError("schema or identity")
        entries, records, end = {}, [], 0
        for entry in index["entries"]:
            name = entry["path"]
            identity = {key: entry[key] for key in ("size", "mtime_ns", "ctime_ns")}
            if (
                not isinstance(name, str)
                or not Path(name).is_absolute()
                or name in entries
                or any(type(value) is not int for value in identity.values())
                or type(entry["offset"]) is not int
                or entry["offset"] != end
                or type(entry["length"]) is not int
                or entry["length"] != identity["size"]
                or entry["length"] <= 0
            ):
                raise ValueError("invalid entry")
            end += entry["length"]
            entries[name] = entry
            records.append(dict(path=name, **identity))
        if (
            not entries
            or index["bytes"] != end
            or (path / "data.bin").stat().st_size != end
            or _key(records) != path.name
        ):
            raise ValueError("invalid length or cache identity")
        return index, entries
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EncodedCacheError(f"invalid encoded cache {path}: {exc}") from exc


class EncodedCacheLease:
    """Own the shared lock, file, mapping and views in that lifetime order."""

    def __init__(self, path, ranges, lock, *, cache_hit=True, evicted=()):
        self.path, self.ranges, self.lock = Path(path), tuple(ranges), lock
        self.cache_hit, self.evicted = cache_hit, list(evicted)
        self.stream = self.mapping = self.buffer = None
        self.views = []
        try:
            self.stream = (self.path / "data.bin").open("rb")
            self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
            self.buffer = memoryview(self.mapping)
            for offset, length in self.ranges:
                if offset < 0 or length <= 0 or offset + length > len(self.buffer):
                    raise EncodedCacheError("cache range outside data.bin")
                self.views.append(self.buffer[offset : offset + length])
        except BaseException:
            self.close()
            raise

    @property
    def descriptor(self):
        return dict(path=str(self.path), ranges=self.ranges)

    @property
    def encoded_bytes(self):
        return self.mapping.size()

    def prefault(self):
        for view in self.views:
            for offset in range(0, len(view), mmap.PAGESIZE):
                view[offset]
            view[-1]

    def close(self):
        for view in self.views:
            view.release()
        self.views.clear()
        if self.buffer is not None:
            self.buffer.release()
            self.buffer = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_cache(descriptor):
    """Reopen the parent's mapping under an independent worker lease."""
    path = Path(descriptor["path"])
    with _root_lock(path.parent):
        _read_index(path)
        lock = _lock(path.parent / ".locks" / path.name, exclusive=False)
        return EncodedCacheLease(path, descriptor["ranges"], lock)


def prepare_cache(filenames, *, cache_dir=None, cache_limit=DEFAULT_CACHE_LIMIT):
    if not sys.platform.startswith("linux"):
        raise EncodedCacheError("decode-only cache requires Linux tmpfs")
    limit = parse_cache_limit(cache_limit)
    paths = [str(Path(name).expanduser().resolve(strict=True)) for name in filenames]
    if not paths:
        raise EncodedCacheError("empty decode-only selection")
    records = [dict(path=name, **_identity(Path(name))) for name in sorted(set(paths))]
    wanted = {entry["path"]: entry for entry in records}
    cache_id = _key(records)
    root = Path(cache_dir or f"/dev/shm/imgread_benchmark-{os.getuid()}").expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = root.resolve()
    info = root.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise EncodedCacheError(
            f"cache root must be owned by this user with mode 0700: {root}"
        )
    _require_tmpfs(root)
    (root / ".locks").mkdir(mode=0o700, exist_ok=True)
    (root / ".used").mkdir(mode=0o700, exist_ok=True)
    with _root_lock(root):
        for leftover in root.glob(".build-*"):
            if leftover.is_symlink() or not leftover.is_dir():
                leftover.unlink()
            else:
                shutil.rmtree(leftover)
        candidates = []
        for path in _datasets(root):
            try:
                index, entries = _read_index(path)
            except EncodedCacheError:
                if not _discard(root, path):
                    raise EncodedCacheError(
                        f"corrupt cache has active leases: {path}"
                    ) from None
                continue
            if _allocated(path) <= limit and all(
                name in entries
                and all(entries[name][key] == value for key, value in record.items())
                for name, record in wanted.items()
            ):
                candidates.append((_allocated(path), path, index, entries))
        if candidates:
            _, path, index, entries = min(
                candidates,
                key=lambda item: (item[1].name != cache_id, item[0], item[1].name),
            )
            evicted = _make_room(root, 0, limit, protected=(path.name,))
            hit = True
        else:
            entries, offset = {}, 0
            for record in records:
                entries[record["path"]] = dict(
                    record, offset=offset, length=record["size"]
                )
                offset += record["size"]
            index = dict(
                schema_version=1,
                cache_id=cache_id,
                bytes=offset,
                entries=list(entries.values()),
            )
            encoded_index = json.dumps(index, sort_keys=True).encode()
            block = os.statvfs(root).f_frsize
            needed = sum(
                ((size + block - 1) // block) * block
                for size in (offset, len(encoded_index))
            )
            if needed > limit:
                raise EncodedCacheError(
                    f"selection needs {needed} bytes, cache limit is {limit}"
                )
            evicted = _make_room(root, needed, limit)
            filesystem = os.statvfs(root)
            available = min(filesystem.f_bavail * filesystem.f_frsize, _memory_budget())
            if needed > available:
                raise EncodedCacheError(
                    f"not enough tmpfs/RAM: need {needed} bytes, available after reserve {available}"
                )
            staging = Path(tempfile.mkdtemp(prefix=".build-", dir=root))
            try:
                with (staging / "data.bin").open("wb") as output:
                    for record in records:
                        source = Path(record["path"])
                        expected = {
                            key: record[key] for key in ("size", "mtime_ns", "ctime_ns")
                        }
                        if _identity(source) != expected:
                            raise EncodedCacheError(
                                f"source changed before copy: {source}"
                            )
                        with source.open("rb") as stream:
                            shutil.copyfileobj(stream, output, length=1024 * 1024)
                        if _identity(source) != expected:
                            raise EncodedCacheError(
                                f"source changed during copy: {source}"
                            )
                    if output.tell() != offset:
                        raise EncodedCacheError(
                            "source size changed during cache build"
                        )
                (staging / "index.json").write_bytes(encoded_index)
                if _usage(root) > limit:
                    raise EncodedCacheError("allocated cache size exceeds the limit")
                path = root / cache_id
                if path.exists() and not _discard(root, path):
                    raise EncodedCacheError(f"cannot replace an active cache: {path}")
                staging.rename(path)
                _read_index(path)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            hit = False
        ranges = [(entries[name]["offset"], entries[name]["length"]) for name in paths]
        lock = _lock(root / ".locks" / path.name, exclusive=False)
        (root / ".used" / path.name).touch()
        return EncodedCacheLease(path, ranges, lock, cache_hit=hit, evicted=evicted)
