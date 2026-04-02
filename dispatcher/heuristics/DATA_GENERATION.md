# Data Generation Guide

This document explains how to build benchmark binaries from the CK Tile engine,
generate benchmark datasets, and manage them for the ML kernel performance
prediction system.

## Overview

The ML heuristic needs benchmark data: measured TFLOPS, latency, and bandwidth
for every (problem shape, kernel config) pair. The tile engine builds one
executable per kernel configuration. Each executable benchmarks a single kernel
on a given problem size and outputs JSON with performance metrics.

```
CK source  -->  CMake configure  -->  ninja build  -->  benchmark binaries
                                                          (4608 per op/dtype/layout)

benchmark binaries  -->  run on GPU  -->  streaming log  -->  parquet dataset
                          (per shape)       (JSON blocks)      (canonical schema)
```

## Prerequisites

- **ROCm**: HIP >= 6.0.3 (for gfx950: HIP >= 6.0.4)
- **Build tools**: CMake >= 3.21, Ninja, HIP-aware clang compiler
- **Python**: 3.10+ with `pandas`, `pyarrow`
- **GPU**: ROCm-capable AMD GPU (MI250X, MI300X, MI355X, etc.)

---

## Part 1: Building Benchmark Binaries from the Tile Engine

If you already have pre-built binaries (e.g., in `/workspace/ck_tile/bin/`),
skip to Part 2. This section explains how to build them from source.

### Step 1: CMake Configure

From the CK repository root:

```bash
cmake -S /workspace/rocm-libraries/projects/composablekernel \
      -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_TARGETS="gfx950" \
      -DGEMM_UNIVERSAL_DATATYPE="fp8" \
      -DGEMM_UNIVERSAL_LAYOUT="rcr" \
      -G Ninja
```

**Key CMake variables:**

| Variable | Default | Description |
|---|---|---|
| `GPU_TARGETS` | (required) | Target GPU architectures. Supported: `gfx90a`, `gfx942`, `gfx950`, `gfx1201`. Semicolon-separated for multiple. |
| `GEMM_UNIVERSAL_DATATYPE` | `"fp8;fp16"` | Data types to build. Options: `fp8`, `fp16`, `bf16`, `bf8`. Semicolon-separated. |
| `GEMM_UNIVERSAL_LAYOUT` | `"rcr;rrr;crr;ccr"` | Layouts to build. Semicolon-separated. |
| `GEMM_UNIVERSAL_CONFIG_FILE` | `"default_config.json"` | Kernel config file (in the `configs/` directory). Controls which tile sizes, warp configs, pipelines, etc. are enumerated. |
| `ENABLE_CCACHE_GEMM_UNIVERSAL` | `OFF` | Enable ccache for faster rebuilds. |

**Example: build only fp8 RCR for gfx950 (fastest, ~4608 kernels):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DGPU_TARGETS="gfx950" \
      -DGEMM_UNIVERSAL_DATATYPE="fp8" \
      -DGEMM_UNIVERSAL_LAYOUT="rcr" \
      -G Ninja
```

**Example: build all dtypes and layouts (slow, ~4608 * 4 * 4 = ~73K kernels):**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DGPU_TARGETS="gfx950" \
      -DGEMM_UNIVERSAL_DATATYPE="fp8;fp16;bf16;bf8" \
      -DGEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" \
      -G Ninja
```

### What happens during configure

1. CMake calls `gemm_universal_instance_builder.py --list_kernels` to enumerate
   all valid kernel configurations from the config JSON.
2. It writes `gemm_universal_kernel_list.txt` (one kernel per line) and
   `gemm_universal_kernel_count.txt` to the build directory.
3. For each kernel, it creates a ninja build target.

### Step 2: Build

```bash
# Build all benchmarks for the configured dtypes/layouts
ninja -C build benchmark_gemm_universal_all

# Or build a specific dtype/layout combo
ninja -C build benchmark_gemm_universal_fp8_rcr

# Or build by pipeline type
ninja -C build benchmark_gemm_universal_compv4_pipeline
ninja -C build benchmark_gemm_universal_mem_pipeline

# Or build a single specific kernel
ninja -C build benchmark_gemm_universal_fp8_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x128_1x4x1_16x16x128
```

**Build time estimates:**
- ~4608 kernels (one dtype, one layout): 1-4 hours depending on CPU cores
- Use `-j <N>` to control parallelism: `ninja -C build -j 32 benchmark_gemm_universal_fp8_rcr`

### Step 3: Verify binaries

Binaries are placed in `build/bin/`:

```bash
ls build/bin/benchmark_gemm_universal_fp8_rcr_* | wc -l
# Expected: 4608 (for default config)

# Test one binary
./build/bin/benchmark_gemm_universal_fp8_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x128_1x4x1_16x16x128 \
    -m=1024 -n=1024 -k=1024 -warmup=3 -repeat=10 -verify=0
```

### Kernel config files

The config files live in:
```
tile_engine/ops/gemm/gemm_universal/configs/
  default_config.json       # Default: full enumeration
  default_ci_config.json    # CI: reduced set for fast testing
  user_provided_config.json # Custom: your own subset
```

To use a custom config:
```bash
cmake ... -DGEMM_UNIVERSAL_CONFIG_FILE="user_provided_config.json"
```

The config controls which tile sizes (e.g., 128x128x64, 256x256x32), warp
configurations (e.g., 2x2x1, 1x4x1), pipelines (compv3, compv4, mem),
schedulers, and other parameters are included in the kernel enumeration.

### Building StreamK / other ops

The same pattern applies to other tile engine ops:

```bash
# StreamK
ninja -C build benchmark_gemm_streamk_fp8_rcr

# Grouped convolution
ninja -C build benchmark_grouped_conv_fwd_fp16_nhwgc
```

Each op has its own instance builder and config directory.

---

## Part 2: Running Benchmarks and Generating Data

## Quick Start

### 1. Run benchmarks for a set of shapes

Each binary accepts `-m=`, `-n=`, `-k=`, `-warmup=`, `-repeat=`, `-verify=` flags
and outputs JSON to stdout:

```bash
/workspace/ck_tile/bin/benchmark_gemm_universal_fp8_rcr_compv3_cshuffle_intrawave_False_False_False_False_128x128x128_1x4x1_16x16x128 \
    -m=1024 -n=1024 -k=1024 -warmup=3 -repeat=10 -verify=0
```

Output:
```json
{
  "name": "gemm_universal_fp8_rcr_compv3_cshuffle_intrawave_...",
  "problem": {
    "split_k": 1, "m": 1024, "n": 1024, "k": 1024,
    "dtype_a": "fp8", "dtype_b": "fp8", ...
  },
  "perf_result": {
    "latency(ms)": 0.04,
    "tflops(TFlops)": 204.60,
    "bandwidth(GB/s)": 624.39
  }
}
```

### 2. Batch generation using provided scripts

**Wide coverage (diverse shapes across all regimes):**
```bash
python3 generate_wide_coverage.py \
    --bin_dir /workspace/ck_tile/bin \
    --out_dir data/wide_coverage \
    --batch_size 25 \
    --warmup 3 --repeat 10
```

**Edge-case dimensions (N=1, K=1, small N/K):**
```bash
python3 generate_edge_dims.py
```

Both scripts write streaming log files that `data_pipeline.py` can parse.

### 3. Parse logs into parquet

```bash
python3 data_pipeline.py <log_file> \
    -o data/my_dataset.parquet \
    --arch gfx950 \
    --capture_hw
```

The `--capture_hw` flag runs `rocminfo` once and injects the GPU hardware
profile (CU count, clock speed, cache sizes, etc.) into every row.

## Canonical Data Schema

Every parquet file follows this schema:

| Column | Type | Description |
|---|---|---|
| `op_type` | str | `gemm_universal`, `gemm_streamk`, etc. |
| `dtype` | str | `fp8`, `fp16`, `bf16`, `bf8` |
| `layout` | str | `rcr`, `rrr`, `crr`, `ccr` |
| `arch` | str | `gfx942`, `gfx950`, etc. |
| `kernel_name` | str | Full kernel identifier |
| `m`, `n`, `k` | int | Problem dimensions |
| `split_k` | int | Split-K factor (1 = standard) |
| `measured_tflops` | float | Ground-truth TFLOPS |
| `latency_ms` | float | Measured latency |
| `bandwidth_gb_s` | float | Measured bandwidth |
| `is_valid` | bool | True if tflops > 0 and latency > 0 |
| `tile_m`, `tile_n`, `tile_k` | int | Tile dimensions |
| `warp_m`, `warp_n`, `warp_k` | int | Warp config |
| `warp_tile_m/n/k` | int | Warp tile dimensions |
| `pipeline` | str | `compv3`, `compv4`, `mem`, etc. |
| `scheduler` | str | `intrawave`, `interwave` |
| `epilogue` | str | `cshuffle`, `default` |
| `pad_m`, `pad_n`, `pad_k` | bool | Padding flags |
| `persistent` | bool | Persistent kernel flag |
| `run_id` | str | Unique collection run identifier |

## Shape Selection Guidelines

Good training data requires diverse shapes. Cover all of these regimes:

### By M dimension (batch size / output rows)
- **M=1**: single-token inference (hardest case for tiling)
- **Tiny M (2-16)**: small batch inference
- **Small M (32-128)**: medium batch
- **Medium M (256-2048)**: large batch / training
- **Large M (4096-20480)**: very large batch

### By N and K dimension
- **N=1**: vector-matrix multiply (degenerate)
- **K=1**: rank-1 update / outer product (degenerate)
- **Small N or K (2-16)**: stress tile efficiency
- **Deep K (K > 4096)**: compute-bound regime
- **Shallow K (K < 256)**: memory-bound regime

### By shape family
- **Square**: M ~ N ~ K (powers of 2)
- **Tall**: M >> N (tall output matrix)
- **Wide**: N >> M (wide output matrix)
- **Deep-K**: K >> M and K >> N

### Special cases
- **Prime dimensions**: 17, 31, 127, 251, 509, 1021, 2039, 4093
  (worst-case for tile alignment, tests padding logic)
- **Non-power-of-2**: 48, 96, 192, 384, 576, 768, 1536, 3072, 4608
  (common in LLM architectures)
- **LLM inference shapes**: DeepSeek, LLaMA-7B, LLaMA-70B MLP/attention dims

### Minimum recommended coverage

For a production-quality model, aim for:
- At least 200 unique (M, N, K) shapes
- At least 10 shapes per shape family
- All kernel configs (4608 for fp8 RCR) run against every shape
- Multiple layouts if training a cross-layout model

## Benchmark Quality Guidelines

### Warmup and repeat
- Minimum `warmup=3`, `repeat=10` for fast iteration
- Production quality: `warmup=5`, `repeat=20` for stable measurements
- The `perf_result` values are averaged over `repeat` iterations

### Noise handling
- Use **median** latency when aggregating multiple runs of the same benchmark
- Flag measurements where coefficient of variation exceeds 10%
- Avoid benchmarking under thermal throttling (check GPU temperature)
- Lock GPU clocks if possible for reproducibility

### Environment metadata
Store with every dataset:
- GPU model and architecture (from `rocminfo`)
- ROCm driver version
- Clock mode (default / locked)
- Git hash of the CK tile engine build (if available)
- Timestamp

## Adding Data for a New Op

To generate benchmark data for a new operation (e.g., `gemm_streamk`):

1. **Build the binaries** using the tile engine:
   ```bash
   ninja -C build benchmark_gemm_streamk_fp8_rcr
   ```

2. **Write a generation script** (or modify `generate_wide_coverage.py`):
   - Change the executable glob pattern to match the new op
   - Add any op-specific CLI flags the binaries need

3. **Run and parse**:
   ```bash
   python3 data_pipeline.py my_streamk_run.log \
       -o data/gemm_streamk_fp8_gfx950.parquet --arch gfx950
   ```

4. **Train**:
   ```bash
   python3 train.py --op gemm_streamk --dtype fp8 --arch gfx950 \
       --data_dir data/ --out_dir models/gemm_streamk_fp8_gfx950
   ```

## Adding Data for a New Layout

Same binaries, same shapes -- just change the layout filter:

```bash
# Build rrr binaries
ninja -C build benchmark_gemm_universal_fp8_rrr

# Generate and parse
# ... (same flow, different bin_dir or executable glob)

# Train a cross-layout model by putting all layouts in the same data_dir
python3 train.py --data_dir data/ --out_dir models/gemm_universal_fp8_gfx950_all_layouts
```

The feature engine includes `layout` as a categorical feature, so one model
can handle all layouts.

## Incremental Data Collection

When you have a trained model and want to add more data:

1. Generate new data (new shapes, new layouts, etc.)
2. Parse into parquet alongside existing data
3. Warm-start from the previous model:
   ```bash
   python3 train.py --data_dir data/ --out_dir models/v2 \
       --warm_start models/v1 \
       --warm_start_n_estimators 200
   ```

This adds 200 new trees on top of the existing model. The feature schema
must match exactly (enforced automatically).

## File Organization

Recommended directory structure:

```
heuristics/
  data/
    gemm_universal_fp8_rcr_gfx950.parquet      # original 108 shapes
    wide_coverage/                               # batch log files
      wide_coverage_batch_001.log
      wide_coverage_batch_002.log
      ...
    edge_dims/                                   # N=1, K=1 edge cases
      edge_dims_batch_001.log
      ...
  models/
    gemm_universal_fp8_gfx950/                  # trained model artifacts
      model_tflops.lgbm
      model_latency.lgbm
      model_bandwidth.lgbm
      feature_spec.json
      train_manifest.json
      cv_metrics_tflops.json
      eval_report.json
      ...
```

## Troubleshooting

### Benchmark binary exits with non-zero code
Some kernel configs are invalid for certain problem sizes (e.g., tile_m=256
with M=16). The data pipeline marks these as `is_valid=False` and they are
filtered out during training. This is expected.

### Edge dims produce very few results
N=1 and K=1 shapes are degenerate -- most kernel configurations have minimum
dimension requirements and will fail or produce zero TFLOPS. The small number
of valid results is still useful (it tells the model which configs work for
these shapes).

### Benchmarks are slow
Each shape requires running all 4608 kernel executables sequentially. At
~0.01s per kernel, that is ~46 seconds per shape. For 700 shapes, expect
~9 hours. Tips:
- Run on a dedicated GPU (no other workloads)
- Use `--batch_size 25` to get incremental output
- Parse and train on partial data while generation continues

### Data from different GPUs / driver versions
Store `run_id` and hardware metadata with each dataset. Training on mixed
data is allowed but not recommended for production models. Filter to a
single `run_id` or `arch` for clean experiments.
