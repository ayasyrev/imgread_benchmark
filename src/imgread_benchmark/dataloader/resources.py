"""External non-atomic CPU/RSS sweeps and conservative epoch association.

CPU uses own counters, 100% = one CPU. RSS sums may double-count shared pages.
No interpolation, extrapolation, baseline subtraction or observer-overhead correction.
"""

from __future__ import annotations

import math
import threading
import time
from copy import deepcopy
from statistics import mean

METHOD = "external psutil own user+system; non-atomic process-tree sweeps; RSS shared pages may be counted repeatedly; sampled peaks may miss transients; observer overhead included"


def preflight_monitor():
    try:
        import psutil

        if not (7, 2) <= tuple(map(int, psutil.__version__.split(".")[:2])) < (8, 0):
            raise ValueError("requires psutil>=7.2,<8")
        process = psutil.Process()
        process.create_time(), process.cpu_times(), process.memory_info()
    except Exception as exc:
        raise ValueError(
            f"monitor-resources unavailable: {exc}; install the monitor extra"
        ) from exc


def identity(process):
    return (process["pid"], process["create_time"])


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _sum_known(values):
    values = [v for v in values if _finite(v)]
    return sum(values) if values else None


def classify_sample(sample, registrations, expected_workers):
    sample = deepcopy(sample)
    roots = {
        (r["pid"], r["create_time"])
        for r in registrations
        if r.get("create_time") is not None
    }
    processes = sample["processes"]
    lookup = {identity(p): p for p in processes}
    for process in processes:
        if process["role"] == "consumer":
            continue
        current = identity(process)
        visited = set()
        while current in lookup and current not in roots and current not in visited:
            visited.add(current)
            parent = lookup[current].get("parent_identity")
            current = tuple(parent) if parent else None
        process["role"] = (
            "worker"
            if identity(process) in roots
            else "descendant"
            if current in roots
            else "excluded_unknown"
        )
    included = [
        p for p in processes if p["role"] in ("consumer", "worker", "descendant")
    ]
    observed_roots = [
        list(identity(p))
        for p in included
        if p["role"] == "worker" and p["status"] == "ok"
    ]
    reasons = list(sample.get("reasons", []))
    if len(observed_roots) < expected_workers:
        reasons.append("expected workers missing or registration delayed")
    if any(p["status"] != "ok" for p in included):
        reasons.append("process counters missing")
    if sample.get("discovery_status") != "complete":
        reasons.append("process discovery incomplete")
    consumer_rss = _sum_known(
        p.get("rss_bytes") for p in included if p["role"] == "consumer"
    )
    worker_rss = _sum_known(
        p.get("rss_bytes") for p in included if p["role"] in ("worker", "descendant")
    )
    if expected_workers == 0 and not any(
        p["role"] in ("worker", "descendant") for p in included
    ):
        worker_rss = 0
    total_rss = _sum_known(p.get("rss_bytes") for p in included)
    if consumer_rss is None or worker_rss is None:
        reasons.append("RSS scope incomplete")
    sample.update(
        expected_worker_count=expected_workers,
        observed_worker_identities=observed_roots,
        expected_worker_identities=[list(r) for r in sorted(roots) if r in lookup],
        totals=dict(
            consumer_rss_bytes=consumer_rss,
            workers_rss_bytes=worker_rss,
            total_rss_bytes=total_rss,
        ),
        completeness="partial" if reasons else "complete",
        reasons=sorted(set(reasons)),
    )
    return sample


def reduce_observations(samples, epochs, registrations=(), expected_workers=0):
    """Pure arithmetic, also used to re-evaluate exported observations."""
    assigned = []
    for raw in samples:
        epoch = next(
            (
                e
                for e in epochs
                if e.start_ns
                <= raw["sweep_start_ns"]
                <= raw["sweep_end_ns"]
                <= e.end_ns
            ),
            None,
        )
        sample = classify_sample(raw, registrations, expected_workers)
        sample["epoch"] = epoch.index if epoch else None
        assigned.append(sample)
    intervals = []
    for previous, current in zip(assigned, assigned[1:]):
        dt = (current["sweep_end_ns"] - previous["sweep_end_ns"]) / 1e9
        reasons = []
        epoch = current["epoch"]
        contained = epoch is not None and previous["epoch"] == epoch
        if not contained:
            reasons.append("cross-boundary")
        if (
            not _finite(dt)
            or dt <= 0
            or previous["sweep_end_ns"] > current["sweep_start_ns"]
        ):
            reasons.append("invalid or overlapping interval")
        pmap = {
            identity(p): p
            for p in previous["processes"]
            if p["role"] in ("consumer", "worker", "descendant")
        }
        cmap = {
            identity(p): p
            for p in current["processes"]
            if p["role"] in ("consumer", "worker", "descendant")
        }
        deltas = []
        for key in pmap.keys() | cmap.keys():
            p, c = pmap.get(key), cmap.get(key)
            if p is None or c is None:
                reasons.append("identity appeared/disappeared (including PID reuse)")
                continue
            counters = [
                p.get("user_seconds"),
                p.get("system_seconds"),
                c.get("user_seconds"),
                c.get("system_seconds"),
            ]
            if not all(_finite(v) for v in counters):
                reasons.append("missing CPU counters")
                continue
            du, ds = counters[2] - counters[0], counters[3] - counters[1]
            if du < 0 or ds < 0:
                reasons.append("CPU counter rollback")
                continue
            deltas.append(dict(pid=key[0], create_time=key[1], cpu_seconds=du + ds))
        if (
            previous["completeness"] != "complete"
            or current["completeness"] != "complete"
        ):
            reasons.append("partial process scope")
        valid_time = (
            _finite(dt)
            and dt > 0
            and previous["sweep_end_ns"] <= current["sweep_start_ns"]
        )
        percent = (
            100 * sum(d["cpu_seconds"] for d in deltas) / dt
            if deltas and valid_time
            else None
        )
        intervals.append(
            dict(
                schema_version=1,
                config_id=current["config_id"],
                start_ns=previous["sweep_end_ns"],
                end_ns=current["sweep_end_ns"],
                interval_seconds=dt if math.isfinite(dt) else None,
                epoch=epoch if contained else None,
                included=contained and percent is not None,
                cpu_percent=percent,
                process_deltas=deltas,
                completeness="missing"
                if percent is None
                else "partial"
                if reasons
                else "complete",
                reasons=sorted(set(reasons)),
            )
        )
    summaries = []
    for epoch in epochs:
        observations = [s for s in assigned if s["epoch"] == epoch.index]
        valid = [i for i in intervals if i["epoch"] == epoch.index and i["included"]]
        duration = sum(i["interval_seconds"] for i in valid)
        rss = {}
        for series in ("consumer", "workers", "total"):
            values = [
                s["totals"][series + "_rss_bytes"] / 2**20
                for s in observations
                if s["totals"][series + "_rss_bytes"] is not None
            ]
            rss[series] = dict(
                mean_mib=mean(values) if values else None,
                observed_peak_mib=max(values) if values else None,
            )
        partial_samples = sum(s["completeness"] != "complete" for s in observations)
        partial_intervals = sum(i["completeness"] != "complete" for i in valid)
        reasons = sorted(
            {r for s in observations for r in s["reasons"]}
            | {r for i in valid for r in i["reasons"]}
        )
        if not observations:
            reasons.append("no in-epoch sweeps")
        if not valid:
            reasons.append("no in-epoch CPU intervals")
        summaries.append(
            dict(
                method=METHOD,
                status="complete"
                if observations
                and valid
                and not partial_samples
                and not partial_intervals
                else "partial",
                cpu_mean_percent=sum(
                    i["cpu_percent"] * i["interval_seconds"] for i in valid
                )
                / duration
                if duration
                else None,
                cpu_max_percent=max((i["cpu_percent"] for i in valid), default=None),
                cpu_interval_seconds=duration,
                cpu_time_coverage=duration / epoch.epoch_seconds
                if epoch.epoch_seconds > 0
                else None,
                complete_process_interval_count=len(valid) - partial_intervals,
                partial_process_interval_count=partial_intervals,
                sample_count=len(observations),
                complete_sample_count=len(observations) - partial_samples,
                partial_sample_count=partial_samples,
                rss=rss,
                reasons=reasons,
            )
        )
    return assigned, intervals, summaries


class ResourceSampler:
    def __init__(
        self, consumer_identity, interval_seconds, config_id, expected_workers
    ):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("sampling interval must be finite and positive")
        import psutil

        self.psutil = psutil
        self.consumer = consumer_identity
        self.interval = interval_seconds
        self.config_id = config_id
        self.expected_workers = expected_workers
        self.samples, self.registrations, self.errors = [], [], []
        self.baseline = None
        self._stop = threading.Event()
        self._thread = None

    def register(self, event):
        self.registrations.append(dict(event))
        if event.get("error"):
            self.errors.append(event["error"])

    def sweep(self):
        psutil = self.psutil
        start = time.perf_counter_ns()
        processes = []
        discovery_status = "complete"
        root = psutil.Process(self.consumer["pid"])
        if root.create_time() != self.consumer["os_create_time"]:
            raise RuntimeError("consumer PID identity changed")
        try:
            candidates = [root, *root.children(recursive=True)]
        except psutil.Error:
            candidates = [root]
            discovery_status = "partial"
        for candidate in candidates:
            row = dict(
                pid=candidate.pid,
                create_time=None,
                parent_identity=None,
                role="consumer" if candidate.pid == root.pid else "candidate",
                counter_timestamp_ns=time.perf_counter_ns(),
                user_seconds=None,
                system_seconds=None,
                rss_bytes=None,
                status="missing",
                reason=None,
            )
            try:
                row["create_time"] = candidate.create_time()
                if candidate.pid != root.pid:
                    parent = candidate.parent()
                    row["parent_identity"] = (
                        [parent.pid, parent.create_time()] if parent else None
                    )
                counters = candidate.cpu_times()
                row.update(
                    user_seconds=counters.user,
                    system_seconds=counters.system,
                    rss_bytes=candidate.memory_info().rss,
                    counter_timestamp_ns=time.perf_counter_ns(),
                    status="ok",
                )
            except psutil.Error as exc:
                row["reason"] = f"{type(exc).__name__}: {exc}"
                discovery_status = "partial"
            processes.append(row)
        return dict(
            schema_version=1,
            config_id=self.config_id,
            sweep_start_ns=start,
            sweep_end_ns=time.perf_counter_ns(),
            epoch=None,
            processes=processes,
            discovery_status=discovery_status,
            reasons=[],
        )

    def start(self):
        try:
            self.baseline = classify_sample(self.sweep(), (), 0)
            self.baseline["method"] = METHOD
        except Exception as exc:
            self.errors.append(f"baseline: {type(exc).__name__}: {exc}")
            return

        def poll():
            while not self._stop.is_set():
                try:
                    self.samples.append(self.sweep())
                except Exception as exc:
                    self.errors.append(f"sampler: {type(exc).__name__}: {exc}")
                    return
                # Wait after each completed sweep: no overlapping sweeps or catch-up bursts.
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                self.errors.append("sampler stop timeout")

    def apply(self, result):
        samples, intervals, summaries = reduce_observations(
            list(self.samples),
            result.epochs,
            list(self.registrations),
            self.expected_workers,
        )
        result.resource_samples, result.resource_intervals = samples, intervals
        for epoch, summary in zip(result.epochs, summaries):
            epoch.resources = summary
        result.resource_status = (
            "partial"
            if self.errors
            or self.baseline is None
            or not summaries
            or any(s["status"] != "complete" for s in summaries)
            else "complete"
        )
        result.preparation["worker_registrations"] = list(self.registrations)
        result.preparation["resource_errors"] = list(self.errors)
        result.warnings.extend(self.errors)
