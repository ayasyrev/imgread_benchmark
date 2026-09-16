"""Immutable file order and source-header validation, never an alternate decoder."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from PIL import Image

from .models import FileManifest, ImageReadError, ManifestEntry, digest


def stat_identity(path):
    info = Path(path).stat()
    return dict(
        size=info.st_size, mtime_ns=info.st_mtime_ns, dev=info.st_dev, ino=info.st_ino
    )


def _precision(path, fmt):
    with open(path, "rb") as stream:
        if fmt == "PNG":
            header = stream.read(33)
            if (
                len(header) != 33
                or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[12:16] != b"IHDR"
                or header[8:12] != b"\0\0\0\r"
            ):
                raise ValueError("invalid PNG IHDR")
            if header[25] not in (0, 2, 4, 6):
                raise ValueError("unsupported PNG color type")
            return header[24]
        if stream.read(2) != b"\xff\xd8":
            raise ValueError("invalid JPEG SOI")
        while True:
            if stream.read(1) != b"\xff":
                raise ValueError("invalid or truncated JPEG marker")
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if not marker or marker in (b"\xda", b"\xd9"):
                raise ValueError("JPEG has no SOF")
            if marker[0] in range(0xD0, 0xD8) or marker == b"\x01":
                continue
            raw = stream.read(2)
            if len(raw) != 2:
                raise ValueError("truncated JPEG length")
            length = struct.unpack(">H", raw)[0]
            if length < 2:
                raise ValueError("invalid JPEG length")
            payload = stream.read(length - 2)
            if len(payload) != length - 2:
                raise ValueError("truncated JPEG header")
            if marker[0] in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            ):
                if not payload:
                    raise ValueError("empty JPEG SOF")
                return payload[0]


def source_header(path, reader="preflight"):
    try:
        with Image.open(path) as image:
            fmt, mode = image.format, image.mode
            frames = getattr(image, "n_frames", 1)
            if fmt not in ("JPEG", "PNG"):
                raise ValueError(f"unsupported source format {fmt}")
            depth = _precision(path, fmt)
            if (
                depth != 8
                or frames != 1
                or mode not in ("RGB", "L", "RGBA", "LA")
                or (fmt == "JPEG" and mode not in ("RGB", "L"))
            ):
                raise ValueError(
                    f"unsupported source: {fmt} mode={mode} bit_depth={depth} frames={frames}"
                )
            return dict(
                format=fmt,
                mode=mode,
                bit_depth=depth,
                width=image.width,
                height=image.height,
                frames=frames,
            )
    except Exception as exc:
        raise ImageReadError(reader, path, exc, "preflight") from exc


def snapshot_files(filenames, *, num_samples=0):
    if type(num_samples) is not int or num_samples < 0:
        raise ValueError("num_samples must be a nonnegative integer")
    paths = [str(Path(p).expanduser().resolve()) for p in filenames]
    paths = paths[:num_samples] if num_samples else paths
    if not paths:
        raise ValueError("Empty image snapshot")
    entries = []
    for path in paths:
        before = stat_identity(path)
        header = source_header(path)
        if stat_identity(path) != before:
            raise ImageReadError(
                "preflight", path, "file changed during snapshot", "preflight"
            )
        entries.append(ManifestEntry(path, header, before))
    return FileManifest(tuple(entries), num_samples, digest(paths))


def discover_manifest(img_path, *, num_samples=0):
    from ..get_img_filenames import get_img_filenames

    return snapshot_files(get_img_filenames(img_path), num_samples=num_samples)


def validate_manifest(manifest, reader):
    FileManifest.from_dict(manifest.to_dict())
    for entry in manifest.entries:
        try:
            if stat_identity(entry.path) != entry.stat:
                raise ValueError("file stat identity changed")
            if source_header(entry.path, reader) != entry.header:
                raise ValueError("source header changed")
        except Exception as exc:
            raise ImageReadError(reader, entry.path, exc, "preflight") from exc


def load_manifest(path, *, num_samples=0):
    manifest = FileManifest.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    if type(num_samples) is not int or num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    if not num_samples:
        return manifest
    entries = manifest.entries[:num_samples]
    return FileManifest(entries, num_samples, digest([e.path for e in entries]))
