# CK ROCProf Tool

GPU performance profiling for Composable Kernel applications using AMD rocprof-compute.

## Quick Start

```bash
# One-time setup
./script/tools/ck-rocprof setup

# Profile executable
cd build
../script/tools/ck-rocprof run baseline ./bin/tile_example_gemm_universal

# Analyze LDS metrics
../script/tools/ck-rocprof analyze baseline

# Compare optimizations
../script/tools/ck-rocprof run optimized ./bin/tile_example_gemm_universal
../script/tools/ck-rocprof compare baseline optimized
```

## Commands

### `setup`
One-time setup: creates Python venv, installs dependencies, configures rocprof-compute.

### `run <name> <executable> [args]`
Profile executable and save results.

```bash
# Basic profiling
ck-rocprof run baseline ./bin/gemm_example

# With arguments
ck-rocprof run large_matrix ./bin/gemm_example -m 8192 -n 8192 -k 4096

# Test filtering
ck-rocprof run unit_test ./bin/test_gemm --gtest_filter="*Fp16*"
```

### `analyze <name> [block]`
Display profiling metrics (default: Block 12 - LDS).

```bash
ck-rocprof analyze baseline        # LDS metrics
ck-rocprof analyze baseline 2      # L2 Cache
ck-rocprof analyze baseline 7      # Instruction Mix
```

### `compare <name1> <name2>`
Side-by-side comparison of two runs.

### `list`
List all profiling runs with size and date.

## Key LDS Metrics (Block 12)

**Target Values:**
- Bank Conflicts/Access: <0.01 (1% conflict rate)
- Bank Conflict Rate: >90% of peak bandwidth

**Critical Metrics:**
- **12.2.9 Bank Conflicts/Access**: Direct conflict measure
  - Baseline (naive): ~0.04 (4% conflicts)
  - Optimized: <0.005 (<0.5% conflicts)
- **12.2.12 Bank Conflict Cycles**: Wasted cycles per kernel
- **12.2.17 LDS Data FIFO Full**: Memory system pressure

## Optimization Workflow

```bash
# 1. Baseline
ck-rocprof run baseline ./bin/my_kernel

# 2. Check conflicts
ck-rocprof analyze baseline
# Look for Bank Conflicts/Access > 0.02

# 3. Optimize code (XOR transforms, padding, etc.)
# ... edit source ...

# 4. Test optimization
ninja my_kernel
ck-rocprof run optimized ./bin/my_kernel

# 5. Verify improvement
ck-rocprof compare baseline optimized
# Target: 8-10x reduction in conflicts
```

## Environment Variables

- `CK_PROFILE_VENV`: Python venv path (default: `/opt/rocprof_venv`)
- `CK_ROCPROF_BIN`: rocprof-compute binary path
- `CK_WORKLOAD_DIR`: Results directory (default: `/workspace/build/workloads`)
- `GPU_TARGET`: Override GPU detection (e.g., `gfx950`, `gfx942`)

## Interpreting Results

**Good Performance:**
```
Bank Conflicts/Access: <0.01
Bank Conflict Rate: >90% of peak
LDS Data FIFO Full: Minimal cycles
```

**Needs Optimization:**
```
Bank Conflicts/Access: >0.02
Bank Conflict Cycles: High MAX values
LDS Data FIFO Full: High memory pressure
```

## Troubleshooting

**"Profiling environment not set up"**
```bash
ck-rocprof setup
```

**"rocprof-compute not found"**
```bash
export CK_ROCPROF_BIN=/custom/path/rocprof-compute
ck-rocprof setup
```

**"Profiling results not found"**
```bash
ck-rocprof list                    # Check available runs
rocminfo | grep gfx               # Verify GPU arch
export GPU_TARGET=gfx950          # Override if needed
```

## Storage Layout

Results stored in `/workspace/build/workloads/<name>/<gpu_arch>/`:
- `SQ_INST_LEVEL_LDS.csv`: LDS metrics
- `pmc_perf.csv`: Performance counters
- `counter_collection.csv`: All metrics

## Technical Details

- **Setup**: Creates isolated Python venv, installs dependencies
- **Profiling**: Runs `rocprof-compute profile --name <name> -- <executable>`
- **Analysis**: Runs `rocprof-compute analyze --path <path> --block <block>`
- **GPU Support**: MI300/MI350 series, auto-detects architecture

## Related Tools

- `ck-docker`: Container management
- `rocprof-compute`: AMD GPU profiler v2
- `rocm-smi`: System monitoring

## License

Copyright (c) Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT
