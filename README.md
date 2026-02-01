# imgread_benchmark
Benchmark for read images with different libs.

## Dataset Management

You can download standard datasets for benchmarking using the `imgread_data` command.

### Imagenette

To download the full Imagenette dataset:
```bash
uv run imgread_data imagenette
```

To download a specific size (e.g., 160px or 320px):
```bash
uv run imgread_data imagenette --size 160
```

Datasets are stored in the `.data/` directory by default.
