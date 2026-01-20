# CK ROCProf Tool

Performance profiling tool for Composable Kernel applications using AMD rocprof-compute.

## Overview

`ck-rocprof` simplifies GPU performance profiling by wrapping rocprofiler-compute with an easy-to-use CLI. It handles Python environment setup, manages profiling runs, and provides convenient comparison tools for optimization workflows.

## Quick Start

```bash
# One-time setup
./script/tools/ck-rocprof setup

# Profile an executable
cd build
../script/tools/ck-rocprof run baseline ./bin/tile_example_gemm_universal

# Analyze results
../script/tools/ck-rocprof analyze baseline

# Compare two runs
../script/tools/ck-rocprof run optimized ./bin/tile_example_gemm_universal
../script/tools/ck-rocprof compare baseline optimized
```

## Commands

### `setup`

One-time setup of the profiling environment.

```bash
ck-rocprof setup
```

**What it does:**
- Creates Python virtual environment at `/opt/rocprof_venv`
- Installs rocprofiler-compute dependencies
- Creates wrapper script for easy invocation

**Requirements:**
- ROCm installed (tested with ROCm 7.0.1)
- Python 3 with venv support
- Write access to `/opt/` directory

---

### `run <name> <executable> [args]`

Profile an executable and save results with a given name.

```bash
ck-rocprof run <name> <executable> [args...]
```

**Arguments:**
- `name` - Unique identifier for this profiling run
- `executable` - Path to executable to profile
- `args` - Optional arguments to pass to the executable

**Examples:**
```bash
# Profile with default arguments
ck-rocprof run baseline ./bin/tile_example_gemm_universal

# Profile with custom arguments
ck-rocprof run custom_size ./bin/tile_example_gemm_universal -m 8192 -n 8192 -k 4096

# Profile test with gtest filter
ck-rocprof run unit_test ./bin/test_ck_tile_gemm --gtest_filter="*Fp16*"
```

**Output:**
- Results stored in `/workspace/build/workloads/<name>/<gpu_arch>/`
- GPU architecture auto-detected (e.g., `gfx950`, `MI350`)

---

### `analyze <name> [block]`

Analyze profiling results for a specific run.

```bash
ck-rocprof analyze <name> [block]
```

**Arguments:**
- `name` - Name of profiling run to analyze
- `block` - Optional block number to display (default: 12 for LDS metrics)

**Examples:**
```bash
# Analyze LDS metrics (Block 12)
ck-rocprof analyze baseline

# Analyze specific block
ck-rocprof analyze baseline 2   # L2 Cache
ck-rocprof analyze baseline 7   # Compute Unit - Instruction Mix
ck-rocprof analyze baseline 12  # Local Data Share (LDS)
```

**Key Metric Blocks:**
- **Block 2**: L2 Cache
- **Block 7**: Compute Unit - Instruction Mix
- **Block 12**: Local Data Share (LDS)
  - 12.1.3: Bank Conflict Rate (% of peak)
  - 12.2.9: Bank Conflicts/Access (conflicts/access)
  - 12.2.12: Bank Conflict (cycles per kernel)
  - 12.2.17: LDS Data FIFO Full Rate

---

### `compare <name1> <name2>`

Compare LDS metrics from two profiling runs side-by-side.

```bash
ck-rocprof compare <name1> <name2>
```

**Arguments:**
- `name1` - First profiling run (typically baseline)
- `name2` - Second profiling run (typically optimized)

**Example:**
```bash
ck-rocprof compare baseline optimized
```

**Output:**
- Side-by-side Block 12 (LDS) metrics
- Useful for validating optimization improvements

---

### `list`

List all available profiling runs.

```bash
ck-rocprof list
```

**Output:**
- Run name
- Disk usage
- Creation date

**Example:**
```bash
$ ck-rocprof list
Available profiling runs:
===========================================
  baseline             [156K, 2026-01-20]
  optimized            [148K, 2026-01-20]
  custom_size          [162K, 2026-01-20]
```

---

## Environment Variables

- `CK_PROFILE_VENV` - Python venv path (default: `/opt/rocprof_venv`)
- `CK_ROCPROF_BIN` - rocprof-compute binary (default: `/opt/rocm-7.0.1/bin/rocprof-compute`)
- `CK_WORKLOAD_DIR` - Workload storage directory (default: `/workspace/build/workloads`)
- `GPU_TARGET` - Override GPU architecture detection (e.g., `gfx950`, `gfx942`)

## Common Workflows

### Optimizing LDS Bank Conflicts

```bash
# 1. Capture baseline metrics
cd build
../script/tools/ck-rocprof run baseline ./bin/tile_example_gemm_universal

# 2. Check baseline bank conflict rate
../script/tools/ck-rocprof analyze baseline
# Look for:
#   - Section 12.2.9: Bank Conflicts/Access (target: <0.01)
#   - Section 12.2.12: Bank Conflict cycles (target: minimize)

# 3. Apply optimization (e.g., XOR transform)
# ... modify code ...

# 4. Rebuild and profile
ninja tile_example_gemm_universal
../script/tools/ck-rocprof run optimized ./bin/tile_example_gemm_universal

# 5. Compare results
../script/tools/ck-rocprof compare baseline optimized

# 6. Verify improvement
# Expected: Bank Conflicts/Access reduced by 8-10x
```

### Profiling Multiple Configurations

```bash
# Profile different tile sizes
ck-rocprof run tile_64x64 ./bin/example -tile 64
ck-rocprof run tile_128x128 ./bin/example -tile 128
ck-rocprof run tile_256x256 ./bin/example -tile 256

# Compare all runs
ck-rocprof list
ck-rocprof compare tile_64x64 tile_128x128
ck-rocprof compare tile_128x128 tile_256x256
```

## Understanding LDS Metrics (Block 12)

### Key Metrics

**12.1.3 - Bank Conflict Rate**
- Percentage of theoretical peak LDS bandwidth achieved
- Lower is worse (conflicts reduce bandwidth)
- Target: >90% of peak

**12.2.9 - Bank Conflicts/Access**
- Average number of bank conflicts per LDS access
- Direct measure of conflict severity
- Target: <0.01 (1% conflict rate)
- Baseline (naive layout): ~0.04 (4% conflicts)
- Optimized (XOR/padding): <0.005 (<0.5% conflicts)

**12.2.12 - Bank Conflict (cycles)**
- Total cycles lost to bank conflicts per kernel
- Shows MAX, MIN, AVG across compute units
- Look at MAX column for worst-case bottleneck
- Target: Minimize (reduce by 8-10x with optimization)

**12.2.17 - LDS Data FIFO Full Rate**
- Cycles stalled due to LDS data FIFO full
- Indicates memory system pressure
- Reduces with bank conflict mitigation

### Interpreting Results

**Good LDS Performance:**
```
Section 12.1 - Speed-of-Light:
  Bank Conflict Rate: Low % of peak

Section 12.2 - LDS Stats:
  Bank Conflicts/Access: Low value (ideally <0.01)
  Bank Conflict (MAX): Minimal cycles
  LDS Data FIFO Full: Low cycles
```

**Poor LDS Performance (needs optimization):**
```
Section 12.1 - Speed-of-Light:
  Bank Conflict Rate: Higher % of peak

Section 12.2 - LDS Stats:
  Bank Conflicts/Access: High value (>0.02 indicates issues)
  Bank Conflict (MAX): Significant cycles lost
  LDS Data FIFO Full: High memory pressure
```

**Typical Improvement After Optimization:**
- Bank Conflicts/Access: Reduced by 10-20x
- Bank Conflict cycles: Significantly reduced
- Overall kernel speedup: Variable depending on workload

## Troubleshooting

### "Error: Profiling environment not set up"

**Solution:** Run `ck-rocprof setup` first

### "Error: rocprof-compute not found"

**Cause:** ROCm not installed or in non-standard location

**Solution:** Set `CK_ROCPROF_BIN` environment variable:
```bash
export CK_ROCPROF_BIN=/path/to/rocm/bin/rocprof-compute
ck-rocprof setup
```

### "Error: Profiling results not found"

**Cause:** Profiling run doesn't exist or wrong GPU architecture

**Solution:**
```bash
# List available runs
ck-rocprof list

# Check GPU architecture
rocminfo | grep gfx

# Override if needed
export GPU_TARGET=gfx950
```

### Python Dependency Conflicts

**Cause:** Ubuntu 24.04 has externally-managed Python environment

**Solution:** The tool automatically creates a venv to avoid conflicts. If setup fails:
```bash
# Manual venv creation
sudo python3 -m venv /opt/rocprof_venv
sudo /opt/rocprof_venv/bin/pip install -r /opt/rocm-7.0.1/libexec/rocprofiler-compute/requirements.txt
```

## Technical Details

### Profiling Workflow

1. **Setup Phase** (`ck-rocprof setup`):
   - Creates isolated Python venv
   - Installs rocprofiler-compute dependencies
   - Creates wrapper script that uses venv Python

2. **Profiling Phase** (`ck-rocprof run`):
   - Detects GPU architecture
   - Runs `rocprof-compute profile --name <name> -- <executable>`
   - Stores results in `workloads/<name>/<gpu_arch>/`

3. **Analysis Phase** (`ck-rocprof analyze`):
   - Runs `rocprof-compute analyze --path <path> --block <block>`
   - Displays formatted metrics from specified block
   - Default: Block 12 (LDS metrics)

### Storage Layout

```
/workspace/build/workloads/
+-- baseline/
|   +-- gfx950/
|       +-- SQ_INST_LEVEL_LDS.csv          # LDS metrics (Block 12)
|       +-- pmc_perf.csv                    # Performance counters
|       +-- counter_collection.csv          # All counters
+-- optimized/
|   +-- gfx950/
|       +-- ...
+-- custom_run/
    +-- MI350/
        +-- ...
```

### Supported GPU Architectures

- AMD Instinct MI300 series (gfx940, gfx941, gfx942)
- AMD Instinct MI350 series (gfx950, gfx951)
- Auto-detection via `rocminfo`
- Manual override via `GPU_TARGET` environment variable

## Related Tools

- `ck-docker` - Docker container management for CK builds
- `rocprof-compute` - AMD's GPU profiling tool (v2)
- `rocm-smi` - ROCm System Management Interface

## References

- [ROCm Profiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [Composable Kernel GitHub](https://github.com/ROCmSoftwarePlatform/composable_kernel)
- [LDS Bank Conflict Optimization Guide](../../../github_formatting_bot_ideas.md)

## Examples

### Example 1: Basic Profiling Session

```bash
# Setup (one-time)
$ ./script/tools/ck-rocprof setup
Setting up rocprof-compute profiling environment...
* Virtual environment created
* Dependencies installed
* Wrapper script created
Setup complete!

# Profile an executable
$ cd build
$ ../script/tools/ck-rocprof run my_profile ./bin/tile_example_gemm_universal
Profiling: ./bin/tile_example_gemm_universal
* Profiling complete
Results saved to: /workspace/build/workloads/my_profile/<gpu_arch>/

# Analyze results
$ ../script/tools/ck-rocprof analyze my_profile
Analyzing: my_profile (Block 12)
[Shows Section 12: Local Data Share (LDS) metrics]
```

### Example 2: Optimization Workflow

```bash
# Profile baseline
$ ck-rocprof run baseline ./bin/my_kernel

# Make code changes
# ... modify source files ...

# Rebuild
$ ninja my_kernel

# Profile optimized version
$ ck-rocprof run optimized ./bin/my_kernel

# Compare results
$ ck-rocprof compare baseline optimized
[Shows side-by-side Block 12 (LDS) metrics for comparison]
```

## License

Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
