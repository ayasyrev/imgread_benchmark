"""Lightweight argparsecfg registration; runtime imports stay in the handler."""

from dataclasses import dataclass
import sys

from ..argparse_compat import field_argument


@dataclass
class DataLoaderCLIConfig:
    img_path: str = field_argument(
        "img_path", default=None, nargs="?", help="Image directory (or use --manifest)"
    )
    reader: str = field_argument("--reader", default="pil-rgb")
    num_samples: int = field_argument("-n", flag="--num-samples", default=0)
    num_workers: int = field_argument(
        flag="--num-workers", default=0, help="0 runs in the consumer process"
    )
    batch_size: int = field_argument(flag="--batch-size", default=32)
    shuffle: bool = field_argument("--shuffle", default=False, action="store_true")
    seed: int = field_argument("--seed", default=0)
    epochs: int = field_argument("--epochs", default=5)
    prefetch_factor: int = field_argument(flag="--prefetch-factor", default=None)
    persistent_workers: bool = field_argument(
        flag="--persistent-workers", default=False, action="store_true"
    )
    pin_memory: bool = field_argument(
        flag="--pin-memory", default=False, action="store_true"
    )
    drop_last: bool = field_argument(
        flag="--drop-last", default=False, action="store_true"
    )
    no_geometry: bool = field_argument(
        flag="--no-geometry", default=False, action="store_true"
    )
    no_warmup: bool = field_argument(
        flag="--no-warmup", default=False, action="store_true"
    )
    monitor_resources: bool = field_argument(
        flag="--monitor-resources", default=False, action="store_true"
    )
    sample_interval_ms: float = field_argument(
        flag="--sample-interval-ms", default=100.0
    )
    output: str = field_argument(
        "--output",
        default=None,
        help="New output directory; existing paths are refused",
    )
    manifest: str = field_argument("--manifest", default=None)
    list_readers: bool = field_argument(
        flag="--list-readers", default=False, action="store_true"
    )


def dataloader(cfg: DataLoaderCLIConfig) -> None:
    from pathlib import Path
    from rich.console import Console
    from .manifest import discover_manifest, load_manifest
    from .models import BenchmarkRunError, DataLoaderConfig, ImageReadError
    from .report import render_result, write_result

    result = None
    exit_code = 0
    try:
        if cfg.list_readers:
            from .readers import list_readers

            for reader in list_readers():
                print(
                    f"{reader.id}: {'available' if reader.available else 'unavailable'} · {reader.version or 'N/A'} · {reader.description}"
                    + (f" · {reader.reason}" if reader.reason else "")
                )
            return
        if bool(cfg.img_path) == bool(cfg.manifest):
            raise ValueError("provide exactly one image directory or --manifest")
        config = DataLoaderConfig(
            reader=cfg.reader,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
            epochs=cfg.epochs,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=cfg.persistent_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            geometry=not cfg.no_geometry,
            warmup=not cfg.no_warmup,
            monitor_resources=cfg.monitor_resources,
            sample_interval_ms=cfg.sample_interval_ms,
        )
        if cfg.output and Path(cfg.output).exists():
            raise FileExistsError(f"output directory already exists: {cfg.output}")
        manifest = (
            load_manifest(cfg.manifest, num_samples=cfg.num_samples)
            if cfg.manifest
            else discover_manifest(cfg.img_path, num_samples=cfg.num_samples)
        )
        from .runner import run_benchmark

        result = run_benchmark(manifest, config)
    except BenchmarkRunError as exc:
        result = exc.result
        exit_code = 130 if (result.error or {}).get("cancelled") else 1
    except ImageReadError as exc:
        print(
            f"stage={exc.stage} reader={cfg.reader} path={exc.path}: {exc.reason}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except (ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt as exc:
        raise SystemExit(130) from exc
    if result is not None:
        Console().print(render_result(result))
        if cfg.output:
            try:
                write_result(result, cfg.output)
            except (OSError, ValueError) as exc:
                print(f"export failed: {exc}", file=sys.stderr)
                raise SystemExit(1) from exc
    if exit_code:
        raise SystemExit(exit_code)
