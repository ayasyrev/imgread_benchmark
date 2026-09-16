"""Deadline-aware JSON transport; the coordinator never writes a blocking pipe."""

import json
import queue
import threading
import time
from collections import deque


class Deadline:
    def __init__(self, expires=None):
        self.expires = expires

    def remaining(self, stage):
        if self.expires is None:
            return None
        remaining = self.expires - time.monotonic()
        if remaining <= 0:
            error = TimeoutError(f"configuration timeout during {stage}")
            error.stage = stage
            raise error
        return remaining


class JsonTransport:
    def __init__(self, process):
        self.process = process
        self.events = queue.Queue()
        self.outgoing = queue.Queue()
        self.stderr = deque(maxlen=1024)
        self.closed = threading.Event()
        self.threads = [
            threading.Thread(target=self._read, args=(process.stdout,), daemon=True),
            threading.Thread(target=self._read_errors, daemon=True),
            threading.Thread(target=self._write, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _read(self, stream):
        try:
            for line in stream:
                self.events.put(line)
        finally:
            self.events.put(None)

    def _read_errors(self):
        for line in self.process.stderr:
            self.stderr.append(line)

    def _write(self):
        try:
            while not self.closed.is_set():
                factory = self.outgoing.get()
                if factory is None:
                    return
                message = factory()
                line = json.dumps(message, allow_nan=False) + "\n"
                if self.closed.is_set():
                    return
                self.process.stdin.write(line)
                self.process.stdin.flush()
        except (OSError, ValueError, TypeError) as exc:
            self.events.put(exc)

    def send(self, message):
        self.send_factory(lambda: message)

    def send_factory(self, factory):
        self.outgoing.put_nowait(factory)

    def receive(self, deadline, stage):
        remaining = deadline.remaining(stage)
        timeout = min(0.1, remaining) if remaining is not None else 0.1
        value = self.events.get(timeout=timeout)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        """Call only after terminating/reaping the entire process group."""
        self.closed.set()
        self.outgoing.put_nowait(None)
        until = time.monotonic() + 2
        for thread in self.threads:
            thread.join(timeout=max(0, until - time.monotonic()))
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("process transport did not stop")
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
