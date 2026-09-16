"""Disposable preflight process, supervised before any consumer is started."""

import json
import os
import sys
from dataclasses import asdict

from .models import DataLoaderConfig, FileManifest, error_details


def main():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def emit(kind, **payload):
        protocol.write(json.dumps(dict(kind=kind, **payload)) + "\n")
        protocol.flush()

    reader_id = None
    try:
        request = json.loads(sys.stdin.readline())
        config = DataLoaderConfig(**request["config"])
        reader_id = config.reader
        emit("phase", stage="preflight.reader")
        from .readers import check_runtime

        try:
            reader = check_runtime(config.reader, config.storage)
            if config.monitor_resources:
                emit("phase", stage="preflight.monitor")
                from .resources import preflight_monitor

                preflight_monitor()
        except (ValueError, ImportError) as exc:
            emit("configuration_error", reason=str(exc))
            return 0
        emit("phase", stage="preflight.manifest")
        from .manifest import validate_manifest

        manifest = FileManifest.from_dict(request["manifest"])
        validate_manifest(manifest, config.reader)
        emit("ready", reader=asdict(reader))
    except BaseException as exc:
        emit("error", error=error_details(exc, reader_id))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
