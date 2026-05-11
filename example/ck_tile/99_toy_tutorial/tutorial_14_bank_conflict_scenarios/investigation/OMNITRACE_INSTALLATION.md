# Omnitrace Installation Guide

## What is Omnitrace?

Omnitrace is AMD's comprehensive profiling and tracing tool for parallel applications. It can:
- Profile CPU and GPU applications
- Collect hardware performance counters
- Generate detailed traces with Perfetto visualization
- Support HIP, OpenCL, C, C++, Fortran, Python applications
- Provide call-stack sampling and binary instrumentation

**Why we need it:** To get detailed LDS bank conflict analysis and understand exactly which instructions cause conflicts.

---

## Your System

- **ROCm Version:** 7.2.0
- **GPU:** AMD MI300X (gfx942)
- **OS:** Linux (Ubuntu/similar)

---

## Installation Methods

### Method 1: Python Install Script (Recommended)

This method doesn't require sudo and installs to a user-specified directory.

#### Step 1: Download the installer

```bash
cd ~
wget https://github.com/ROCm/omnitrace/releases/latest/download/omnitrace-install.py
```

#### Step 2: Run installation for ROCm 7.2

```bash
python3 ./omnitrace-install.py --prefix /opt/omnitrace --rocm 7.2
```

**If you don't have sudo for /opt, install to your home:**
```bash
python3 ./omnitrace-install.py --prefix ~/omnitrace --rocm 7.2
```

#### Step 3: Setup environment

Add to your `~/.bashrc`:
```bash
# Omnitrace setup
source ~/omnitrace/share/omnitrace/setup-env.sh
```

Or manually:
```bash
export PATH="~/omnitrace/bin:$PATH"
export LD_LIBRARY_PATH="~/omnitrace/lib:$LD_LIBRARY_PATH"
```

Then reload:
```bash
source ~/.bashrc
```

#### Step 4: Verify installation

```bash
omnitrace-avail --help
omnitrace-sample --version
```

---

### Method 2: Pre-built Packages (.sh installer)

Visit the [releases page](https://github.com/ROCm/omnitrace/releases) and download the `.sh` installer for ROCm 7.2.

```bash
# Download (check releases page for exact filename)
wget https://github.com/ROCm/omnitrace/releases/download/v1.11.2/omnitrace-1.11.2-ubuntu-22.04-ROCm-70200-PAPI-OMPT-Python3.sh

# Make executable
chmod +x omnitrace-1.11.2-ubuntu-22.04-ROCm-70200-PAPI-OMPT-Python3.sh

# Run installer
./omnitrace-1.11.2-ubuntu-22.04-ROCm-70200-PAPI-OMPT-Python3.sh --prefix=~/omnitrace --exclude-subdir
```

---

### Method 3: Build from Source (Advanced)

If you need the latest version or custom build:

```bash
git clone https://github.com/ROCm/omnitrace.git
cd omnitrace
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=~/omnitrace \
      -DROCM_PATH=/opt/rocm-7.2.0 \
      ..
make -j$(nproc)
make install
```

---

## Quick Start: Profiling Your XOR Test

### 1. Call-Stack Sampling (Easiest)

Profile your application with sampling (doesn't require recompilation):

```bash
omnitrace-sample -- ./04_row_major_xor_asm
```

This generates:
- `omnitrace-*/` directory with results
- Open `perfetto-trace.proto` in https://ui.perfetto.dev

### 2. Binary Instrumentation (More Detailed)

For detailed function-level profiling:

```bash
omnitrace-instrument -o 04_row_major_xor_asm.inst -- ./04_row_major_xor_asm
./04_row_major_xor_asm.inst
```

### 3. Collect Hardware Counters

To see LDS bank conflicts and other GPU metrics:

```bash
export OMNITRACE_USE_ROCM_SMI=ON
export OMNITRACE_USE_ROCPROFILER=ON
omnitrace-sample -- ./04_row_major_xor_asm
```

---

## Useful Environment Variables

```bash
# Enable ROCProfiler for GPU metrics
export OMNITRACE_USE_ROCPROFILER=ON

# Enable specific hardware counters
export OMNITRACE_ROCPROFILER_METRICS="SQ_LDS_BANK_CONFLICT,SQ_WAVES,FETCH_SIZE"

# Output directory
export OMNITRACE_OUTPUT_PATH=./omnitrace-results

# Verbose output
export OMNITRACE_VERBOSE=3

# Enable timeline trace
export OMNITRACE_USE_PERFETTO=ON
```

---

## Analyzing LDS Bank Conflicts

### Configuration for Bank Conflict Analysis

Create a config file `omnitrace.cfg`:
```bash
# GPU metrics
OMNITRACE_USE_ROCPROFILER = true
OMNITRACE_ROCPROFILER_METRICS = SQ_LDS_BANK_CONFLICT

# Enable kernel tracing
OMNITRACE_USE_ROCTRACER = true

# Output format
OMNITRACE_USE_PERFETTO = true
OMNITRACE_PERFETTO_COMBINED_TRACES = true
```

Run with config:
```bash
omnitrace-sample -c omnitrace.cfg -- ./04_row_major_xor_asm
```

### View Results

1. **Perfetto (Timeline):**
   - Open https://ui.perfetto.dev
   - Load `omnitrace-*/perfetto-trace.proto`
   - See GPU kernel execution with metrics

2. **Text Output:**
   ```bash
   cat omnitrace-*/wall_clock.txt
   cat omnitrace-*/sampling.txt
   ```

3. **Hardware Counters:**
   ```bash
   # Check if LDS_BANK_CONFLICT was collected
   grep -r "SQ_LDS_BANK_CONFLICT" omnitrace-*/
   ```

---

## Troubleshooting

### Problem: "omnitrace-sample: command not found"

**Solution:**
```bash
source ~/omnitrace/share/omnitrace/setup-env.sh
```

### Problem: "ROCm not found"

**Solution:**
Check ROCm path is correct:
```bash
export ROCM_PATH=/opt/rocm-7.2.0
```

### Problem: No GPU metrics collected

**Solution:**
Ensure rocprofiler is enabled:
```bash
export OMNITRACE_USE_ROCPROFILER=ON
export HSA_TOOLS_LIB=/opt/rocm-7.2.0/lib/librocprofiler64.so
```

### Problem: Permission denied

**Solution:**
Install to home directory instead of /opt:
```bash
python3 ./omnitrace-install.py --prefix ~/omnitrace --rocm 7.2
```

---

## Advanced: Compare XOR vs No-XOR

Profile both versions side-by-side:

```bash
# Profile without XOR
omnitrace-sample -o results-no-xor -- ./01_row_major

# Profile with XOR
omnitrace-sample -o results-xor -- ./04_row_major_xor_asm

# Compare results
diff results-no-xor/wall_clock.txt results-xor/wall_clock.txt
```

---

## Documentation Links

- **Official Docs:** [ROCm Omnitrace Documentation](https://rocm.docs.amd.com/projects/omnitrace/en/latest/)
- **GitHub:** [ROCm/omnitrace](https://github.com/ROCm/omnitrace)
- **Releases:** [Download page](https://github.com/ROCm/omnitrace/releases)
- **Tutorial (2024):** [Cray User Group Tutorial](https://hackmd.io/@sfantao/cug2024-omnitrace)

---

## Next Steps

1. **Install Omnitrace** using Method 1 (Python script)
2. **Verify** it works with `omnitrace-avail`
3. **Profile your XOR test** with `omnitrace-sample`
4. **Analyze results** in Perfetto UI
5. **Compare** bank conflict metrics between XOR and non-XOR versions

This will give you much more detailed insight into **exactly which instructions cause bank conflicts** and **how the XOR transformation affects them**.
