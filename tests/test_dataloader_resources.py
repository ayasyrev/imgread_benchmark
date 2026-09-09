from copy import deepcopy

import pytest

from imgread_benchmark.dataloader.models import EpochResult
from imgread_benchmark.dataloader.resources import reduce_observations


def sample(seconds, cpu, consumer=100, workers=20, pid=2):
    return dict(
        schema_version=1,
        config_id="test",
        sweep_start_ns=int(seconds * 1e9),
        sweep_end_ns=int(seconds * 1e9),
        discovery_status="complete",
        processes=[
            dict(
                pid=1,
                create_time=1,
                role="consumer",
                parent_identity=None,
                user_seconds=cpu,
                system_seconds=0,
                children_user=10000,
                rss_bytes=consumer * 2**20,
                status="ok",
            ),
            dict(
                pid=pid,
                create_time=1,
                role="candidate",
                parent_identity=[1, 1],
                user_seconds=0,
                system_seconds=0,
                rss_bytes=workers * 2**20,
                status="ok",
            ),
            dict(
                pid=99,
                create_time=1,
                role="candidate",
                parent_identity=[1, 1],
                user_seconds=10000,
                system_seconds=0,
                rss_bytes=999 * 2**20,
                status="ok",
            ),
        ],
    )


def epoch(end=8):
    return EpochResult.measured(0, 0, end * 10**9, 10, 3, 0, "a", "b")


def test_weighted_cpu_scope_and_rss():
    samples = [sample(1, 0), sample(2, 1), sample(5, 7, 80, 70)]
    assigned, intervals, summaries = reduce_observations(
        samples, [epoch()], [dict(pid=2, create_time=1)], 1
    )
    summary = summaries[0]
    assert summary["cpu_mean_percent"] == 175
    assert summary["cpu_max_percent"] == 200
    assert summary["cpu_time_coverage"] == 0.5
    assert summary["rss"]["total"]["observed_peak_mib"] == 150
    assert assigned[0]["processes"][2]["role"] == "excluded_unknown"
    _, _, pair = reduce_observations(
        samples[1:], [epoch()], [dict(pid=2, create_time=1)], 1
    )
    assert pair[0]["rss"]["total"]["mean_mib"] == 135
    assert len(intervals) == 2


@pytest.mark.parametrize("change", ["missing", "reuse", "rollback", "denied"])
def test_partial_counters(change):
    first, second = sample(1, 0), sample(2, 1)
    if change == "missing":
        second["processes"].pop(1)
    elif change == "reuse":
        second["processes"][1]["create_time"] = 2
    elif change == "rollback":
        first["processes"][1]["user_seconds"] = 2
    else:
        second["processes"][1].update(
            user_seconds=None, status="missing", rss_bytes=None
        )
    _, intervals, summaries = reduce_observations(
        [first, second], [epoch()], [dict(pid=2, create_time=1)], 1
    )
    assert intervals[0]["cpu_percent"] == 100
    assert intervals[0]["completeness"] == "partial"
    assert summaries[0]["cpu_time_coverage"] == 1 / 8


def test_boundary_and_empty():
    samples = [sample(1, 0), sample(2, 1), sample(5, 7)]
    e = epoch(3)
    _, intervals, summaries = reduce_observations(samples, [e], (), 0)
    assert intervals[1]["included"] is False
    assert "cross-boundary" in intervals[1]["reasons"]
    assert summaries[0]["cpu_mean_percent"] == 100
    _, _, empty = reduce_observations([], [e])
    assert empty[0]["cpu_mean_percent"] is None
    assert empty[0]["rss"]["total"]["observed_peak_mib"] is None


def test_non_atomic_sweep_must_fit():
    value = sample(1, 0)
    value["sweep_start_ns"] = -1
    assigned, _, summaries = reduce_observations([value], [epoch()])
    assert assigned[0]["epoch"] is None
    assert summaries[0]["sample_count"] == 0


def test_first_interval_and_invalid_time():
    first = sample(1, 0)
    second = deepcopy(first)
    _, intervals, summaries = reduce_observations([first, second], [epoch()])
    assert intervals[0]["cpu_percent"] is None
    assert summaries[0]["cpu_mean_percent"] is None


def test_descendant_scope():
    first, second = sample(1, 0), sample(2, 1)
    for item in (first, second):
        item["processes"][2]["parent_identity"] = [2, 1]
    second["processes"][2]["user_seconds"] += 1
    assigned, _, summaries = reduce_observations(
        [first, second], [epoch()], [dict(pid=2, create_time=1)], 1
    )
    assert assigned[0]["processes"][2]["role"] == "descendant"
    assert summaries[0]["cpu_mean_percent"] == 200


def test_sampler_failure_preserves_success(monkeypatch):
    from imgread_benchmark.dataloader.models import BenchmarkResult, DataLoaderConfig
    from imgread_benchmark.dataloader.resources import ResourceSampler
    from imgread_benchmark.dataloader import snapshot_files
    from tests.dataloader_helpers import make_images

    # exercised by the real runner tests too; here force a polling failure deterministically
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        result = BenchmarkResult(
            "id",
            "uuid",
            snapshot_files(make_images(root, 1)),
            DataLoaderConfig().effective(),
            status="success",
            epochs=[epoch()],
        )
    sampler = ResourceSampler({}, 0.01, "id", 0)
    monkeypatch.setattr(
        sampler, "sweep", lambda: (_ for _ in ()).throw(RuntimeError("injected"))
    )
    sampler.start()
    sampler.stop()
    sampler.apply(result)
    assert result.status == "success"
    assert result.resource_status == "partial"
    assert "injected" in result.warnings[0]


def test_midstream_sampler_failure_keeps_observations(monkeypatch, tmp_path):
    from imgread_benchmark.dataloader import snapshot_files
    from imgread_benchmark.dataloader.models import BenchmarkResult
    from imgread_benchmark.dataloader.resources import ResourceSampler
    from tests.dataloader_helpers import make_images

    result = BenchmarkResult(
        "id",
        "uuid",
        snapshot_files(make_images(tmp_path, 1)),
        {},
        status="success",
        epochs=[epoch()],
    )
    sampler = ResourceSampler({}, 0.001, "id", 0)
    values = iter([sample(0, 0), sample(1, 1)])

    def sweep():
        try:
            return next(values)
        except StopIteration as exc:
            raise RuntimeError("poll failed after observation") from exc

    monkeypatch.setattr(sampler, "sweep", sweep)
    sampler.start()
    sampler._thread.join(timeout=2)
    sampler.stop()
    sampler.apply(result)
    assert not sampler._thread.is_alive()
    assert result.status == "success" and result.resource_status == "partial"
    assert sampler.baseline and len(result.resource_samples) == 1
    assert "poll failed" in result.warnings[0]


def test_missing_consumer_rss_is_not_zero():
    item = sample(1, 0)
    item["processes"][0].update(rss_bytes=None, status="missing")
    _, _, summaries = reduce_observations([item], [epoch()], (), 0)
    assert summaries[0]["rss"]["total"]["mean_mib"] is None
    assert summaries[0]["rss"]["workers"]["mean_mib"] == 0
