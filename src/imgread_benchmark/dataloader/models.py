"""Versioned, dependency-light contracts for the CPU DataLoader benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

READER_IDS = ("pil-rgb", "torchvision-rgb", "cv2-rgb", "cv2-bgr-cvtcolor")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def error_details(exc, reader=None):
    """Serialize a failure before teardown can replace its traceback or context."""
    return dict(
        stage=getattr(exc, "stage", "execution"),
        reader=getattr(exc, "reader", reader),
        path=getattr(exc, "path", None),
        reason=f"{type(exc).__name__}: {exc}",
        cancelled=isinstance(exc, KeyboardInterrupt),
    )


@dataclass(frozen=True)
class DataLoaderConfig:
    reader: str = "pil-rgb"
    num_workers: int = 0
    batch_size: int = 32
    shuffle: bool = False
    seed: int = 0
    epochs: int = 5
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    pin_memory: bool = False
    drop_last: bool = False
    geometry: bool = True
    warmup: bool = True
    monitor_resources: bool = False
    sample_interval_ms: float = 100.0

    def __post_init__(self):
        for name in ("num_workers", "batch_size", "seed", "epochs"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in (
            "shuffle",
            "persistent_workers",
            "pin_memory",
            "drop_last",
            "geometry",
            "warmup",
            "monitor_resources",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.reader not in READER_IDS:
            raise ValueError(f"Unknown reader {self.reader!r}; choose {READER_IDS}")
        if self.num_workers < 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError(
                "workers must be nonnegative; batch_size and epochs must be positive"
            )
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if self.prefetch_factor is not None:
            if (
                type(self.prefetch_factor) is not int
                or self.prefetch_factor <= 0
                or self.num_workers == 0
            ):
                raise ValueError(
                    "explicit prefetch_factor requires workers > 0 and a positive integer"
                )
        if self.persistent_workers and not self.num_workers:
            raise ValueError("persistent_workers requires workers > 0")
        if (
            not isinstance(self.sample_interval_ms, (int, float))
            or not math.isfinite(self.sample_interval_ms)
            or self.sample_interval_ms <= 0
        ):
            raise ValueError("sample_interval_ms must be finite and positive")

    @property
    def effective_prefetch_factor(self):
        return (self.prefetch_factor or 2) if self.num_workers else None

    def effective(self):
        return {**asdict(self), "prefetch_factor": self.effective_prefetch_factor}

    def validate_count(self, count):
        if count <= 0 or (self.drop_last and count < self.batch_size):
            raise ValueError("configuration would deliver zero images")


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    header: dict
    stat: dict


@dataclass(frozen=True)
class FileManifest:
    entries: tuple[ManifestEntry, ...]
    requested_n: int
    selection_id: str
    schema_version: int = 1

    @property
    def paths(self):
        return tuple(e.path for e in self.entries)

    @property
    def selected_n(self):
        return len(self.entries)

    def to_dict(self):
        return {**asdict(self), "selected_n": self.selected_n}

    @classmethod
    def from_dict(cls, value):
        try:
            return cls._parse(value)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid manifest schema: {exc}") from exc

    @classmethod
    def _parse(cls, value):
        from pathlib import Path

        if value.get("schema_version") != 1:
            raise ValueError("Unsupported manifest schema_version")
        entries = tuple(ManifestEntry(**e) for e in value["entries"])
        result = cls(entries, value["requested_n"], value["selection_id"])
        if not entries or type(result.requested_n) is not int or result.requested_n < 0:
            raise ValueError("Invalid manifest selection")
        if any(
            not Path(p).is_absolute() or str(Path(p).resolve()) != p
            for p in result.paths
        ):
            raise ValueError("Manifest paths must be absolute and normalized")
        if (
            value.get("selected_n") != len(entries)
            or digest(list(result.paths)) != result.selection_id
        ):
            raise ValueError("Invalid manifest count or selection_id")
        for entry in entries:
            if not isinstance(entry.header, dict) or set(entry.header) != {
                "format",
                "mode",
                "bit_depth",
                "width",
                "height",
                "frames",
            }:
                raise ValueError("Invalid manifest source header")
            if set(entry.stat) != {"size", "mtime_ns", "dev", "ino"} or any(
                type(v) is not int for v in entry.stat.values()
            ):
                raise ValueError("Invalid manifest stat identity")
        if result.requested_n and result.selected_n > result.requested_n:
            raise ValueError("Manifest selected count exceeds requested count")
        return result


@dataclass(frozen=True)
class ReaderInfo:
    id: str
    description: str
    available: bool
    reason: str | None = None
    version: str | None = None


@dataclass
class EpochResult:
    index: int
    kind: str
    status: str
    start_ns: int
    end_ns: int
    epoch_seconds: float
    images_delivered: int
    batches_delivered: int
    images_dropped: int
    order_id: str
    delivered_order_id: str
    ms_per_image: float | None
    images_per_second: float | None
    batches_per_second: float | None
    resources: dict | None = None

    @classmethod
    def measured(
        cls,
        index,
        start,
        end,
        images,
        batches,
        dropped,
        order_id,
        delivered_order_id,
        status="success",
    ):
        seconds = (end - start) / 1e9
        valid = status == "success" and images > 0 and batches > 0 and seconds > 0
        return cls(
            index,
            "first" if index == 0 else "subsequent",
            status,
            start,
            end,
            seconds,
            images,
            batches,
            dropped,
            order_id,
            delivered_order_id,
            1000 * seconds / images if valid else None,
            images / seconds if valid else None,
            batches / seconds if valid else None,
        )


@dataclass
class BenchmarkResult:
    config_id: str
    execution_id: str
    manifest: FileManifest
    config: dict
    status: str = "failed"
    error: dict | None = None
    reader: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    consumer: dict = field(default_factory=dict)
    isolation: str = "fresh_subprocess"
    preparation: dict = field(default_factory=dict)
    baseline: dict | None = None
    epochs: list[EpochResult] = field(default_factory=list)
    pinning: dict = field(default_factory=dict)
    resource_status: str = "off"
    resource_samples: list[dict] = field(default_factory=list)
    resource_intervals: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: int = 1

    @property
    def late_epoch_seconds_median(self):
        values = [
            e.epoch_seconds
            for e in self.epochs
            if e.index > 0 and e.status == "success"
        ]
        return median(values) if values else None

    def to_dict(self):
        return {
            **asdict(self),
            "manifest": self.manifest.to_dict(),
            "late_epoch_seconds_median": self.late_epoch_seconds_median,
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported result schema_version")
        value.pop("late_epoch_seconds_median", None)
        value["manifest"] = FileManifest.from_dict(value["manifest"])
        value["epochs"] = [EpochResult(**e) for e in value["epochs"]]
        return cls(**value)


class BenchmarkRunError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(str(result.error))


class ImageReadError(ValueError):
    def __init__(self, reader, path, reason, stage="decode"):
        self.reader, self.path, self.reason, self.stage = (
            reader,
            str(path),
            str(reason),
            stage,
        )
        super().__init__(f"reader={reader} path={path}: {reason}")
