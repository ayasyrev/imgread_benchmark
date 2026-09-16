import json
import os
from pathlib import Path

import pytest

from imgread_benchmark import encoded_cache as cache


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_require_tmpfs", lambda root: None)
    monkeypatch.setattr(cache, "_memory_budget", lambda: 2**40)
    return tmp_path / "cache"


def sources(tmp_path, count=3):
    paths = []
    for i in range(count):
        path = tmp_path / f"{i}.bin"
        path.write_bytes(bytes([i + 1]) * 100)
        paths.append(path)
    return paths


@pytest.mark.parametrize(
    "value, expected",
    [("17", 17), ("2KiB", 2048), ("3MiB", 3 * 1024**2), ("2GiB", 2 * 1024**3), (7, 7)],
)
def test_limit_units(value, expected):
    assert cache.parse_cache_limit(value) == expected


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5GiB", "2GB", True, 0])
def test_limit_rejects_invalid(value):
    with pytest.raises(ValueError):
        cache.parse_cache_limit(value)


def test_reuse_order_duplicates_and_no_source_reads(tmp_path, cache_root, monkeypatch):
    paths = sources(tmp_path)
    with cache.prepare_cache(paths, cache_dir=cache_root) as lease:
        assert not lease.cache_hit
        original = lease.path
        assert lease.encoded_bytes == 300
        assert all(view.readonly for view in lease.views)
    real_open = Path.open

    def guarded(path, *args, **kwargs):
        assert path not in paths, "cache hit read a source file"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with cache.prepare_cache(
        [paths[2], paths[0], paths[2]], cache_dir=cache_root
    ) as lease:
        assert lease.cache_hit and lease.path == original
        assert [bytes(view) for view in lease.views] == [
            bytes([3]) * 100,
            bytes([1]) * 100,
            bytes([3]) * 100,
        ]
        lease.prefault()
    assert original.exists()


def test_selected_change_rebuilds_unselected_change_does_not(tmp_path, cache_root):
    paths = sources(tmp_path)
    with cache.prepare_cache(paths, cache_dir=cache_root) as lease:
        original = lease.path
    paths[2].write_bytes(b"changed")
    with cache.prepare_cache(paths[:2], cache_dir=cache_root) as lease:
        assert lease.cache_hit and lease.path == original
    paths[0].write_bytes(b"new selected content")
    with cache.prepare_cache(paths[:2], cache_dir=cache_root) as lease:
        assert not lease.cache_hit and lease.path != original
        assert bytes(lease.views[0]) == b"new selected content"


def test_lru_and_active_lease_limit(tmp_path, cache_root):
    paths = sources(tmp_path)
    with cache.prepare_cache(
        paths[:1], cache_dir=cache_root, cache_limit=16384
    ) as first:
        first_path = first.path
        with cache.prepare_cache(
            paths[1:2], cache_dir=cache_root, cache_limit=16384
        ) as second:
            second_path = second.path
        with cache.prepare_cache(
            paths[2:], cache_dir=cache_root, cache_limit=16384
        ) as third:
            assert first_path.exists() and not second_path.exists()
            assert second_path.name in third.evicted
            with pytest.raises(cache.EncodedCacheError, match="active caches"):
                cache.prepare_cache(paths[1:2], cache_dir=cache_root, cache_limit=8192)
            assert bytes(first.views[0]) == bytes([1]) * 100
    with cache.prepare_cache(
        paths[2:], cache_dir=cache_root, cache_limit=8192
    ) as lease:
        assert lease.cache_hit and not first_path.exists()


def test_smaller_limit_builds_compact_subset(tmp_path, cache_root):
    paths = sources(tmp_path, 2)
    paths[1].write_bytes(b"b" * 10000)
    with cache.prepare_cache(paths, cache_dir=cache_root) as lease:
        original = lease.path
    with cache.prepare_cache(
        paths[:1], cache_dir=cache_root, cache_limit=8192
    ) as lease:
        assert not lease.cache_hit and lease.path != original
        assert not original.exists()


def test_corrupt_inactive_cache_rebuilt_active_cache_preserved(tmp_path, cache_root):
    paths = sources(tmp_path, 1)
    with cache.prepare_cache(paths, cache_dir=cache_root) as lease:
        path = lease.path
        (path / "index.json").write_text("{broken")
        with pytest.raises(cache.EncodedCacheError, match="active leases"):
            cache.prepare_cache(paths, cache_dir=cache_root)
    with cache.prepare_cache(paths, cache_dir=cache_root) as lease:
        assert not lease.cache_hit
        assert bytes(lease.views[0]) == bytes([1]) * 100


def test_failed_copy_releases_lock_and_publishes_nothing(
    tmp_path, cache_root, monkeypatch
):
    paths = sources(tmp_path, 1)
    copy = cache.shutil.copyfileobj

    def fail(*args, **kwargs):
        raise OSError("injected ENOSPC")

    monkeypatch.setattr(cache.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="ENOSPC"):
        cache.prepare_cache(paths, cache_dir=cache_root)
    assert not cache._datasets(cache_root)
    assert not list(cache_root.glob(".build-*"))
    monkeypatch.setattr(cache.shutil, "copyfileobj", copy)
    with cache.prepare_cache(paths, cache_dir=cache_root):
        pass


def test_change_during_copy_fails(tmp_path, cache_root, monkeypatch):
    paths = sources(tmp_path, 1)
    copy = cache.shutil.copyfileobj

    def change(source, output, **kwargs):
        copy(source, output, **kwargs)
        paths[0].write_bytes(b"replaced")

    monkeypatch.setattr(cache.shutil, "copyfileobj", change)
    with pytest.raises(cache.EncodedCacheError, match="changed during copy"):
        cache.prepare_cache(paths, cache_dir=cache_root)
    assert not cache._datasets(cache_root)


def test_insufficient_ram_and_empty_payload(tmp_path, cache_root, monkeypatch):
    paths = sources(tmp_path, 1)
    monkeypatch.setattr(cache, "_memory_budget", lambda: 0)
    with pytest.raises(cache.EncodedCacheError, match="tmpfs/RAM"):
        cache.prepare_cache(paths, cache_dir=cache_root)
    paths[0].write_bytes(b"")
    with pytest.raises(cache.EncodedCacheError, match="nonempty"):
        cache.prepare_cache(paths, cache_dir=cache_root)


def test_stale_build_cleanup_and_invalid_range(tmp_path, cache_root):
    paths = sources(tmp_path, 1)
    with cache.prepare_cache(paths, cache_dir=cache_root) as lease:
        path = lease.path
    leftover = cache_root / ".build-interrupted"
    leftover.mkdir()
    (leftover / "partial").write_bytes(b"partial")
    index = json.loads((path / "index.json").read_text())
    index["entries"][0]["offset"] = 999999
    (path / "index.json").write_text(json.dumps(index))
    with cache.prepare_cache(paths, cache_dir=cache_root) as lease:
        assert not lease.cache_hit
        assert not leftover.exists()
        descriptor = lease.descriptor
        with cache.open_cache(descriptor) as other:
            assert bytes(other.views[0]) == bytes(lease.views[0])
    assert cache._allocated(path) == 8192


def test_reject_nonprivate_root(tmp_path, cache_root):
    paths = sources(tmp_path, 1)
    cache_root.mkdir(mode=0o755)
    os.chmod(cache_root, 0o755)
    with pytest.raises(cache.EncodedCacheError, match="0700"):
        cache.prepare_cache(paths, cache_dir=cache_root)


@pytest.mark.parametrize("version", [1, 2])
def test_memory_budget_accounts_for_parent_cgroup_headroom(
    monkeypatch, tmp_path, version
):
    group = tmp_path / "groups"
    (group / "child").mkdir(parents=True)
    names = (
        ("memory.max", "memory.current")
        if version == 2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    (group / names[0]).write_text(str(2 * 1024**3))
    (group / names[1]).write_text(str(1024**3))
    (group / "child" / names[0]).write_text("max" if version == 2 else str(2**63 - 1))
    (group / "child" / names[1]).write_text("0")
    read_text = Path.read_text

    def fake_read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemTotal: 8388608 kB\nMemAvailable: 6291456 kB\n"
        if str(path) == "/proc/self/cgroup":
            return "0::/child" if version == 2 else "5:memory:/child"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read)
    monkeypatch.setattr(
        cache,
        "_mounts",
        lambda: [(group, "/", f"cgroup{2 if version == 2 else ''}", "memory")],
    )
    assert cache._memory_budget() == 512 * 1024**2


def test_rejects_non_tmpfs(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "_mounts", lambda: [(Path("/"), "/", "ext4", "rw")])
    with pytest.raises(cache.EncodedCacheError, match="tmpfs"):
        cache._require_tmpfs(tmp_path)
