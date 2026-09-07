"""One schema for Python, Rich output and lossless private JSON exports."""

import json
import os
from pathlib import Path

from rich.table import Table

from .models import BenchmarkResult, canonical_json


def write_private(path, text):
    with open(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(text)


def write_result(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    data = result.to_dict()
    for field, filename in (
        ("resource_samples", "samples.jsonl"),
        ("resource_intervals", "intervals.jsonl"),
    ):
        records = data.pop(field)
        write_private(
            output_dir / filename,
            "".join(canonical_json(record) + "\n" for record in records),
        )
        data[field + "_file"] = filename
    write_private(
        output_dir / "manifest.json", canonical_json(result.manifest.to_dict()) + "\n"
    )
    # Result is the completion marker: written last, never overwrite a prior export.
    write_private(output_dir / "result.json", canonical_json(data) + "\n")


def read_result(output_dir):
    output_dir = Path(output_dir)
    data = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    for field, filename in (
        ("resource_samples", "samples.jsonl"),
        ("resource_intervals", "intervals.jsonl"),
    ):
        if data.pop(field + "_file") != filename:
            raise ValueError("Invalid resource stream reference")
        data[field] = [
            json.loads(line)
            for line in (output_dir / filename).read_text(encoding="utf-8").splitlines()
        ]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if data["manifest"] != manifest:
        raise ValueError("Export manifest mismatch")
    return BenchmarkResult.from_dict(data)


def _number(value):
    return "N/A" if value is None else f"{value:.6g}"


def render_result(result):
    config = result.config["effective"]
    title = f"{config['reader']} · {result.status} · workers={config['num_workers']} batch={config['batch_size']} · {result.config_id[:12]}"
    table = Table(title=title)
    columns = [
        "Epoch",
        "Status",
        "Seconds",
        "N",
        "B",
        "Dropped",
        "ms/image",
        "images/s",
    ]
    monitored = config["monitor_resources"]
    if monitored:
        columns += [
            "CPU mean/max %",
            "CPU coverage",
            "RSS mean/peak MiB",
            "Samples/intervals",
            "Resources",
        ]
    for column in columns:
        table.add_column(column)
    for epoch in result.epochs:
        row = [
            f"{epoch.index} ({epoch.kind})",
            epoch.status,
            _number(epoch.epoch_seconds),
            str(epoch.images_delivered),
            str(epoch.batches_delivered),
            str(epoch.images_dropped),
            _number(epoch.ms_per_image),
            _number(epoch.images_per_second),
        ]
        if monitored:
            resource = epoch.resources or {}
            rss = resource.get("rss", {}).get("total", {})
            row += [
                f"{_number(resource.get('cpu_mean_percent'))}/{_number(resource.get('cpu_max_percent'))}",
                _number(resource.get("cpu_time_coverage")),
                f"{_number(rss.get('mean_mib'))}/{_number(rss.get('observed_peak_mib'))}",
                f"{resource.get('sample_count', 0)}/{resource.get('complete_process_interval_count', 0) + resource.get('partial_process_interval_count', 0)}",
                resource.get("status", "partial"),
            ]
        table.add_row(*row)
    caption = f"Late epoch median: {_number(result.late_epoch_seconds_median)} s. Pin requested={config['pin_memory']}, observed={result.pinning.get('status', 'not_observed')}."
    if monitored:
        from .resources import METHOD

        baseline = (result.baseline or {}).get("totals", {}).get("total_rss_bytes")
        caption += f" Baseline RSS: {_number(baseline / 2**20 if baseline is not None else None)} MiB. Interval: {config['sample_interval_ms']} ms; resources={result.resource_status}. {METHOD}."
    if result.error:
        caption += f" Error: {result.error}"
    table.caption = caption
    return table
