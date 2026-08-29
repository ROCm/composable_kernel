# Grouped Convolution Tile Engine

Benchmarking harness for grouped convolution kernels via the CK dispatcher's pipelined JIT compilation.

Covers all three variants -- **forward**, **backward-data**, **backward-weight** -- across the suffix-aware pipeline pool (compv3 / compv4 / compv5 / mem, intrawave / interwave, optional `dsb` / `si` suffixes) for 2D and 3D shapes.

This directory is purely a benchmarking and sweep tool. ML kernel-selection heuristics, training, and validation live in `dispatcher/heuristics/` (see [Related Documentation](#related-documentation)).

## Directory Layout

```
grouped_conv/
  grouped_conv_full_benchmark.py     Orchestrator: enumerate kernels x problems, JIT compile, benchmark
  grouped_conv_instance_builder.py   Kernel enumeration from JSON trait config
  run_one_grouped_conv_kernel.py     Subprocess worker (one kernel, fresh GPU context)
  README.md                          This file
  configs/                           Kernel trait configurations
    forward_bf16.json                  Forward bf16 (compv3/v4/v5)
    bwd_data.json                      Backward data (compv3 / mem)
    bwd_weight.json                    Backward weight (compv3 / mem)
  problems/                          Problem datasets (registry keys consumed by --problems)
    forward_2d.py / forward_3d.py
    bwd_data_2d.py / bwd_data_3d.py
    bwd_weight_2d.py / bwd_weight_3d.py
    *_test_validation.py               Small unseen-shape subsets
    validation_holdout.py              VALIDATION_PROBLEMS (300 forward shapes)
```

## Quick Start

```bash
# Count kernels matching a trait config without compiling
python grouped_conv_instance_builder.py configs/forward_bf16.json --arch gfx950 --count-only

# List kernel names
python grouped_conv_instance_builder.py configs/forward_bf16.json --arch gfx950 --list

# Smoke benchmark: forward 2D on the validation subset
python grouped_conv_full_benchmark.py \
  --variant forward \
  --problems forward_2d_test_validation \
  --workers 256 \
  --output sweep_forward_smoke.csv

# Full sweep: all forward kernels x all forward-2D problems
python grouped_conv_full_benchmark.py \
  --variant forward \
  --problems forward_2d \
  --workers 256 \
  --output sweep_forward_2d.csv

# Backward data / weight sweeps
python grouped_conv_full_benchmark.py --variant bwd_data   --problems bwd_data_2d   --output sweep_bwd_data.csv
python grouped_conv_full_benchmark.py --variant bwd_weight --problems bwd_weight_2d --output sweep_bwd_weight.csv
```

The benchmark always starts fresh and overwrites `--output`. Move or rename the file beforehand if you need to keep prior results.

## How It Works

### Kernel Enumeration

```
JSON trait config (variant + allowed pipelines / wave modes / suffixes)
  --> grouped_conv_instance_builder.py
    --> dispatcher/codegen/grouped_config_rules.py (tile + suffix-aware pool)
      --> list of GroupedConvKernelConfig
        --> optional --filter expression
```

The pipeline rules in `dispatcher/codegen/grouped_config_rules.py` are the single source of truth for the kernel pool (tile sizes, wave modes, pipeline variants, `dsb` / `si` suffixes). The instance builder reads a JSON trait allow-list and produces the cartesian product of legal configurations.

### Benchmark Pipeline

```
grouped_conv_full_benchmark.py (orchestrator)
  |-- grouped_conv_instance_builder.py    enumerate kernel configs
  |-- Build phase                          codegen -> hipcc -> link .so (serial; avoids fork + GPU init issues)
  '-- Benchmark phase                      one subprocess per kernel batch
        '-- run_one_grouped_conv_kernel.py
              '-- GpuGroupedConvRunner     fresh HIP context per problem
```

Key design choices:

1. **Subprocess isolation** -- a fresh HIP context per kernel batch avoids cumulative driver/device leaks during long sweeps.
2. **Serial GPU access** -- accurate timing, no contention.
3. **Path-only build in the main process** -- the orchestrator never initializes the GPU runtime, so `fork()` after codegen is safe.
4. **Batch size ~20 kernels/subprocess** -- empirically a good throughput/overhead tradeoff.

> The `--workers` flag controls codegen/compile parallelism for the build phase. Benchmarking itself is serial per device.

## JSON Config Format

```json
{
  "variant": "forward",
  "trait_config": {
    "data_type":    {"values": ["bf16"]},
    "pipeline":     {"values": ["compv3", "compv4", "compv5"]},
    "wave_mode":    {"values": ["intrawave", "interwave"]},
    "ndim_spatial": {"values": [2, 3]}
  }
}
```

Allowed keys mirror `GroupedConvKernelConfig` fields. See `dispatcher/codegen/grouped_config_rules.py` for the full schema.

### Filtering examples

```bash
# Only large tiles on compv5
python grouped_conv_instance_builder.py configs/forward_bf16.json \
  --arch gfx950 \
  --filter "c.tile_n >= 128 and c.pipeline == 'compv5'" --list

# Export the resolved kernel list to JSON
python grouped_conv_instance_builder.py configs/forward_bf16.json \
  --arch gfx950 --export-json kernels.json
```

## Problem Registry

`--problems` accepts **only registry keys**, not file paths. The keys are wired in `grouped_conv_full_benchmark.py`. Current keys:

| Key                              | Direction      | Notes                                    |
|----------------------------------|----------------|------------------------------------------|
| `forward_2d` / `forward_3d`      | forward        | Full training-grade problem sets         |
| `bwd_data_2d` / `bwd_data_3d`    | backward data  | Full training-grade problem sets         |
| `bwd_weight_2d` / `bwd_weight_3d`| backward wgt   | Full training-grade problem sets         |
| `*_test_validation`              | per direction  | Small unseen-shape subsets               |
| `validation_holdout`             | forward        | 300 shapes (250 2D + 50 3D)              |

Adding a new subset requires both a `problems/<name>.py` file and a registry entry in `grouped_conv_full_benchmark.py`.

Each problem module exposes a list of dataclasses with fields `N, C, K, G, Hi, Wi[, Di], Y, X[, Z], stride_h, stride_w[, stride_d], pad_h, pad_w[, pad_d]` and optional `dilation_*`.

## Output CSV Schema

```
kernel, problem_idx, N, C, K, G, [Di,] Hi, Wi, [Z,] Y, X,
        [stride_d,] stride_h, stride_w,
        [pad_d,]    pad_h,    pad_w,
        latency_ms, tflops, non_zero
```

`non_zero` is a sanity flag (output checksum != 0). Failed launches are written with `latency_ms=N/A` and `tflops=0`.

## Hardware

- Validated on AMD Instinct MI355X (gfx950).
- Datatypes: bf16 (primary), fp16, fp32.
- Pipelines: compv3 / compv4 / compv5 (forward), compv3 / mem (backward).
- Schedulers: intrawave, interwave (with optional `dsb`, `si` suffixes).

### GPU access caveat (this host)

On the dev host the device files have non-default GIDs (`/dev/kfd` GID 506, `/dev/dri/renderD144` GID 109). If `hipMalloc` returns code 100 (`hipErrorOutOfMemory`) on every allocation, it is a permissions issue, not VRAM exhaustion. Launch the benchmark via `sudo -u sshuser bash -lc '...'` so the process tree picks up `kfdhost`, `renderhost`, and `video` groups.

## Related Documentation

Anything ML-heuristic-related has been moved out of this directory:

- **ML training pipeline & models**: `dispatcher/heuristics/README.md`
- **ML vs oracle comparison & validation**: `dispatcher/heuristics/validation/grouped_conv/`
  - `validate_ml_vs_oracle.py` -- run trained predictor over a problem set and compare against oracle CSVs produced by this harness.
  - `compare_ml_vs_oracle.py` -- post-hoc comparison of oracle + ML prediction CSVs (efficiency, top-k, scatter plot).
- **Dispatcher Python API**: `dispatcher/python/`
- **End-to-end examples**: `dispatcher/examples/grouped_conv/`