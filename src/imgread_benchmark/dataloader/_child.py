"""Private JSON-lines subprocess entry point; stdout is reserved for protocol v1."""

import json
import os
import signal
import sys
import threading

from .models import DataLoaderConfig, FileManifest


def main():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    lock = threading.Lock()
    config_id = None

    def emit(event, **payload):
        with lock:
            protocol.write(
                json.dumps(
                    dict(
                        protocol_version=1, config_id=config_id, event=event, **payload
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
            protocol.flush()

    def cancel(signum=None, frame=None):
        raise KeyboardInterrupt("configuration cancelled")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        request = json.loads(sys.stdin.readline())
        if (
            request.get("protocol_version") != 1
            or request.get("event") != "run_request"
        ):
            raise ValueError("invalid run_request")
        config_id = request["config_id"]
        config = DataLoaderConfig(**request["config"])
        manifest = FileManifest.from_dict(request["manifest"])

        def wait_go():
            value = json.loads(sys.stdin.readline())
            if value.get("event") != "go" or value.get("config_id") != config_id:
                raise KeyboardInterrupt("go not received")

            def watch_cancel():
                # stdin is control-only; EOF also cancels an orphaned consumer.
                os.read(sys.stdin.fileno(), 4096)
                os.kill(os.getpid(), signal.SIGINT)

            threading.Thread(target=watch_cancel, daemon=True).start()

        from .engine import execute

        execute(manifest, config, request["execution_id"], emit, wait_go)
    except BaseException as exc:
        emit(
            "run_error",
            error=dict(
                stage=getattr(exc, "stage", "execution"),
                reader=getattr(
                    exc,
                    "reader",
                    locals().get("config", None).reader
                    if "config" in locals()
                    else None,
                ),
                path=getattr(exc, "path", None),
                reason=f"{type(exc).__name__}: {exc}",
                cancelled=isinstance(exc, KeyboardInterrupt),
            ),
        )
        emit("done", status="failed")
        return 1
    emit("done", status="success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
