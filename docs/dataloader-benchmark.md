# CPU DataLoader image benchmark

`imgread_benchmark dataloader` measures CPU batches from one immutable file list.
It compares `pil-rgb`, `torchvision-rgb`, `cv2-rgb` and `cv2-bgr-cvtcolor`.
There is no model, training loop, GPU transfer, decoded cache or automatic reader fallback.

## Installation and supported runtime

Use Python 3.12 or 3.13 on Linux. The qualified runtime pair is exactly
`torch==2.10.0`, `torchvision==0.25.0`; build suffixes such as `+cpu` are retained
in results. The dataloader extra also installs `opencv-python-headless==4.13.0.90`.
The monitor extra supplies `psutil>=7.2,<8` (7.2.2 in the lock).

```bash
uv sync --extra dataloader --extra monitor
uv run --extra dataloader imgread_benchmark dataloader --list-readers
```

Listing reports availability, version and failure reason for all four paths in
isolated probes. With torch/torchvision installed separately, PIL and torchvision
can work without OpenCV. Missing dependencies produce exit 2 before timing.
Base installation and legacy commands do not require torch or psutil. The new
execution mode requires Linux process groups; help and listing remain available
on other platforms.

## CLI and examples

Each output path must be new; exports never overwrite an existing run.

```bash
uv run --extra dataloader --extra monitor imgread_benchmark dataloader tests/test_imgs --reader pil-rgb --num-workers 0 --batch-size 2 --epochs 3 --output /tmp/imgread-dataloader-example-a
uv run --extra dataloader --extra monitor imgread_benchmark dataloader --manifest /tmp/imgread-dataloader-example-a/manifest.json --reader cv2-bgr-cvtcolor --num-workers 2 --persistent-workers --shuffle --monitor-resources --batch-size 4 --epochs 3 --output /tmp/imgread-dataloader-example-b
```

`--num-workers 0` runs loading in the isolated consumer. The legacy benchmark's
`--nw 0` still means all CPUs; the new command has no `--nw` alias.

| Flag | Default | Meaning |
|---|---|---|
| `--reader` | `pil-rgb` | One explicit decode path |
| `-n`, `--num-samples` | 0 | First N files; 0 means all |
| `--num-workers` | 0 | Nonnegative worker count |
| `--batch-size` | 32 | Positive batch size |
| `--shuffle`, `--seed` | false, 0 | Epoch sampler; seed in `[0,2**63)` |
| `--epochs` | 5 | Positive epoch count |
| `--prefetch-factor` | unspecified | Effective None with 0 workers, otherwise 2 |
| `--persistent-workers` | false | Keep workers between epochs of this run |
| `--pin-memory` | false | Request CPU pinned batches; observed separately |
| `--drop-last` | false | Drop the last incomplete batch |
| `--no-geometry` | false | Preserve source H/W after the common RGB boundary |
| `--no-warmup` | false | Disable the byte-only warmup |
| `--monitor-resources` | false | Enable external CPU/RSS observations |
| `--sample-interval-ms` | 100.0 | Finite positive sampling interval |
| `--output` | none | New private export directory |
| `--manifest` | none | Restore an exact saved snapshot |
| `--list-readers` | false | Probe capabilities without benchmarking |

Exactly one directory or manifest is required outside listing. Discovery retains
its existing unsorted order, then applies N without deduplication. A loaded
manifest preserves order and duplicates; N slices it without rebuilding file
identities. Empty input and zero-output drop_last configurations are errors.
Explicit prefetch with zero workers and persistence with zero workers are errors.
For N=10, batch=4 the measured counts are N=10/B=3, or N=8/B=2/dropped=2 with
`--drop-last`. Native sizes must match within each batch; use `--no-geometry
--batch-size 1` for mixed image sizes.

## Python API

```python
from pathlib import Path
from imgread_benchmark.dataloader import (
    DataLoaderConfig, snapshot_files, run_benchmark, write_result, load_manifest,
)

manifest = snapshot_files([Path("photo-a.png"), Path("photo-b.jpg")])
result = run_benchmark(manifest, DataLoaderConfig(epochs=3, batch_size=2))
write_result(result, Path("/tmp/imgread-api-first"))
restored = load_manifest("/tmp/imgread-api-first/manifest.json")
comparison = run_benchmark(restored, DataLoaderConfig(
    reader="torchvision-rgb", epochs=3, batch_size=2,
    num_workers=2, persistent_workers=True, monitor_resources=True,
))
```

`ImageListDataset(manifest, reader_id, geometry=True)` yields `(image, 0)` and
can be used directly in an ordinary PyTorch DataLoader. `ImageTransform` and
`EpochSampler` are public. `run_benchmark` optionally accepts a positive
`timeout_seconds` budget for one invocation, including preparation. Runtime errors
raise `BenchmarkRunError` with `.result` containing completed and failed epochs.
Configuration/dependency errors raise ValueError. API and CLI use the same schemas.

## Image and order contracts

Input is ordinary 8-bit RGB/L JPEG/PNG or RGBA/LA PNG. Grayscale is repeated to
three RGB channels; alpha is discarded without compositing. EXIF orientation
and ICC transforms are not applied. Header validation covers the entire selected
snapshot, including drop_last entries. Source 16-bit, CMYK, palette, 1-bit,
animated/multipage, BMP/TIFF and malformed headers fail explicitly. PNG IHDR and
JPEG SOF precision are checked before decoder execution; Pillow header probing
does not decode pixels and is never a fallback. Real format is checked regardless
of extension.

PIL uses `Image.open().convert("RGB")` and `np.asarray`. Torchvision uses only
`decode_image(..., RGB, apply_exif_orientation=False)` and keeps its Tensor.
OpenCV direct RGB uses `IMREAD_COLOR_RGB | IMREAD_IGNORE_ORIENTATION`; the BGR
variant uses `IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION` then `cvtColor(BGR2RGB)`.
Missing flags or `imread` returning None cause failure.

Every reader enters one CPU uint8 RGB boundary. HWC ndarrays become CHW Tensors;
read-only, nonpositive-stride or unaligned arrays get a writable C-order copy.
Positive-stride views are allowed. Both input types share the same contiguous
policy and the same torchvision Tensor resize (short side 256, bilinear,
antialias=True), then center crop 224×224. Long side uses integer truncation;
crop uses Python round/ties-to-even. Output is uint8, unscaled 0–255; there is no
float pipeline or Normalize. JPEG implementations need not match bit for bit.

Each epoch uses an independent CPU generator seeded with `(seed+epoch) % 2**63`
for `randperm`, or ordered range when shuffle is off. Worker RNG uses a separate
DataLoader generator. Results include hashes of the full sampler order and the
delivered prefix, including the effect of drop_last.

## Timing, isolation and pinning

Every API invocation launches a fresh interpreter and one CPU DataLoader. The
consumer is fresh with monitoring both on and off. Workers use spawn and ordered
batch delivery. Imports, header checks, byte warmup in 1 MiB chunks, Dataset and
loader creation happen before the epoch timer. No decoded dataset is retained.
Warmup can populate OS caches; isolation does not reset caches or system load.

The monotonic timer starts immediately before `iter(loader)` and ends immediately
after exhaustion. Nonpersistent workers' normal shutdown is included. There are
no model computations, extra warmup epochs or per-batch progress messages.
Each epoch publishes elapsed seconds, actual N/B/dropped, ms/image, images/s and
batches/s. The first epoch is separate; late median uses completed subsequent
epochs only. Failed epochs retain partial counts and elapsed time with null rates.

Consumer and workers use one torch intra/inter-op thread and, for OpenCV, one cv2
thread. OMP/MKL/OpenBLAS env values are scoped to the consumer. Effective settings,
versions, OS, CPU/affinity, geometry and process identities are exported. No system
thread settings or caller environment are modified.

Pinning is checked on the last real batch after the timer; status distinguishes
not_requested, not_observed, observed_pinned and observed_unpinned. This does not
guarantee all batches are pinned or promise acceleration. Pinning is not GPU transfer.

Private torch teardown is confined to `close_loader` for the supported version.
The coordinator drains protocol/stdout and diagnostics/stderr throughout execution.
Cancellation and failure trigger cleanup: 10 seconds graceful, 5 seconds TERM,
then KILL and a final bounded check. The grace period begins on the final or
failed epoch, a run error, completion, or EOF; waiting for stdout or process exit
uses that same deadline. A stalled teardown fails even without a configuration
timeout. Further runs are refused if a prior group
cannot be cleaned up. Main-thread SIGTERM and Ctrl-C use the same cleanup path.
Read/iteration failures are published before teardown. Their reader/path context
remains the primary error if shutdown also fails; later errors are retained in
`error.secondary_errors`.

The final table and exported `delivery_summary` explicitly mark incomplete
reading for the selected reader. They show selected entries per epoch, confirmed
batch deliveries versus the planned total, completed epochs, and intentional
`drop_last` exclusions separately. A missing epoch report contributes no confirmed
deliveries; that does not prove that its workers read zero files. Prefetch and
partial batches mean the exact unread-file count cannot be inferred from delivered
batches. Failed-epoch rates remain N/A, and the report shows captured warnings.

## Resource measurements

Monitoring is off by default: no sampler, psutil import, registration queue or
relay is created. When enabled, an external coordinator records the prepared
consumer's RSS baseline before epoch zero. Samples, prior results and summaries
are kept outside the consumer. Each worker registers PID plus OS creation time;
only the consumer, registered workers and their descendants are included. The
observer, its unrelated children and multiprocessing resource tracker are excluded.

CPU uses each process's own user+system counters, never children counters.
100% means one fully busy CPU; totals may exceed 100%. Mean is weighted by each
valid interval's duration. Both complete sweeps and their interval must fit one
epoch. Cross-boundary intervals are retained but excluded from mean/max/coverage;
there is no interpolation. The first observation alone gives no CPU rate.

CPU time coverage is valid interval seconds / epoch seconds. It is separate from
process completeness: consumer-only partial counters can cover time while workers
are missing. PID reuse, disappearing/late workers, counter rollback, AccessDenied,
short epochs and sampling failures retain null/partial labels. A sampling failure
does not turn successful iteration into failed timing or complete resource data.

RSS is the sum of observations in a non-atomic sweep. Consumer, workers and total
have mean and observed peak in MiB (bytes / 2**20). Total peak is the maximum of
per-sweep sums, never the sum of individual maxima. Shared pages can be counted
multiple times. Short-lived processes and peaks between samples may be missed.
Baseline is separate; it is not subtracted or interpreted as exact allocations.
Observer overhead remains in wall time. Compare monitoring mode and interval
alongside results; no universal performance ranking is implied.

## Export, errors and qualification

Exports use schema version 1 and UTF-8 JSON with no NaN; missing metrics are null.
`manifest.json` stores absolute normalized paths, source headers, stat identities,
counts and selection hash. The hash identifies the path list, not image contents.
Stat identity is checked before every decode. Adversarial rewrites preserving all
stat fields are not guaranteed to be detected.

`result.json` contains config, versions, process identity, timings, baseline,
summaries and references to `samples.jsonl` and `intervals.jsonl`. Empty resource
streams are written with monitoring off. `read_result(directory)` restores the
Python result losslessly. Directories use 0700 and files 0600. Result JSON is
written last, so incomplete exports have no completion marker. No automatic upload
or publication occurs; exports can contain personal absolute paths.

CLI exit codes: 0 success/listing; 2 invalid config/dependency/platform; 1 image,
worker, execution or export failure; 130 cancellation. Reader/path/stage details
are retained where known; unknown paths are null.

The accepted [implementation plan](plans/20260907_143308_dataloader-benchmark.plan.md)
defines the fixed post-review qualification: 16 configurations and A/B/B/A, each
three epochs on ten deterministic RGB PNGs. `tests/dataloader_smoke.py` supports
`prepare`, `run`, `verify`, requires an approved exact clean SHA, validates hashes,
and enforces resource and time budgets. The host must have at least 8 GiB available
RAM, four logical CPUs, 2 GiB artifact storage and 256 MiB free shared memory.
No retries, matrix reductions or speed thresholds are used. Artifacts remain under
`/tmp/imgread-dataloader-acceptance/20260907_143308_dataloader-benchmark/<SHA>`;
/tmp does not promise retention across system cleanup. A missing host prerequisite
means qualification is incomplete, even when development tests pass.
