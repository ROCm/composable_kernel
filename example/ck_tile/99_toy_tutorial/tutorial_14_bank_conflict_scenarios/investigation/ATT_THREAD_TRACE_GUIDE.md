# AMD Thread Trace (ATT) - Assembly-Level Instruction Profiling Guide

## What is ATT (Advanced Thread Trace)?

ATT captures **every single instruction** executed by the GPU at the assembly level, showing:
- Exact instruction sequence
- Execution latency per instruction
- Which instructions cause stalls/conflicts
- Hardware counter values per instruction
- **Assembly code annotations**

This is the tool you need to see **exactly which `ds_read_u16` instructions cause LDS bank conflicts!**

---

## Your System Status

✅ **You have ATT support installed:**
- `/opt/rocm-7.2.0/lib/rocprofiler/libatt_plugin.so` (trace decoder)
- `/opt/rocm-7.2.0/libexec/rocprofiler/att/att.py` (decoder script)
- Built into `rocprofv3`

✅ **Your GPU (MI300X/gfx942) supports ATT**

---

## Quick Start: Profile Your XOR Kernel

### Step 1: Run ATT Profiling

```bash
cd /home/aghamari/composable_kernel/example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios

# Basic ATT profiling
rocprofv3 \
  --att \
  --hip-trace \
  --kernel-trace \
  --att-target-cu 0 \
  --att-simd-select 0xF \
  --output-format pftrace \
  -o att_xor \
  -- ./bin/aa_tutorial_14_04_row_major_xor
```

**What this does:**
- `--att` - Enable Advanced Thread Trace
- `--att-target-cu 0` - Profile CU 0 (where your kernel runs)
- `--att-simd-select 0xF` - Profile all 4 SIMDs (bitmask 0xF = 1111)
- Output includes `.att` trace files + Perfetto timeline

### Step 2: View Results

**Option A: Perfetto Timeline (Quick View)**
1. Open https://ui.perfetto.dev
2. Load `att_xor_results.pftrace`
3. See instruction-level timeline with assembly annotations

**Option B: ATT Decoder (Detailed Assembly View)**
```bash
# Find the ATT trace file
ls att_xor/*.att

# Decode to human-readable format
/opt/rocm-7.2.0/libexec/rocprofiler/att/att.py att_xor/<kernel_name>.att

# Or convert to CSV for analysis
/opt/rocm-7.2.0/libexec/rocprofiler/att/att_to_csv.py att_xor/<kernel_name>.att -o xor_instructions.csv
```

---

## Advanced: ATT with LDS Conflict Counters

To correlate LDS bank conflicts with specific instructions:

```bash
# Create metrics file
cat > att_lds_metrics.txt << 'EOF'
pmc: SQ_LDS_BANK_CONFLICT
pmc: SQ_INSTS_LDS
pmc: SQ_WAVE_CYCLES
pmc: SQ_WAVES
EOF

# Run ATT with hardware counters
rocprofv3 \
  --att \
  --hip-trace \
  --kernel-trace \
  -i att_lds_metrics.txt \
  --att-target-cu 0 \
  --att-simd-select 0xF \
  --att-buffer-size 512 \
  --output-format pftrace \
  -o att_with_counters \
  -- ./bin/aa_tutorial_14_04_row_major_xor
```

---

## ATT Options Explained

### CU and SIMD Selection

```bash
--att-target-cu 0              # CU ID to profile (default: 1)
--att-simd-select 0xF          # SIMD bitmask (0xF = all 4 SIMDs)
                               # For gfx942: 0x1=SIMD0, 0x2=SIMD1, 0x4=SIMD2, 0x8=SIMD3
```

### Buffer Size

```bash
--att-buffer-size 256          # Buffer in MB (default: 256)
                               # Increase if trace is truncated
```

### Kernel Filtering

```bash
--kernel-include-regex "ProductionTransposeKernel"  # Only trace this kernel
--kernel-exclude-regex "flush_cache"                # Exclude helper kernels
```

### Multiple Kernels

```bash
--att-consecutive-kernels 5    # Profile first 5 kernel launches
```

---

## Understanding ATT Output

### 1. Perfetto Timeline View

When you load `.pftrace` in Perfetto, you'll see:
- **Timeline tracks** for each instruction
- **Assembly code** in tooltip on hover
- **Latency/stalls** visualized as gaps
- **Hardware counters** overlaid

### 2. ATT Trace File (.att)

Binary format containing:
- Program counter (PC) values
- Instruction addresses
- Wave IDs
- Execution times
- Hardware counter samples

### 3. Decoded CSV Output

Example columns:
```
PC, Instruction, Opcode, Latency, HitCount, Bank_Conflicts, ...
```

Each row = one instruction execution

---

## Analyzing Your XOR Bank Conflicts

### What to Look For:

1. **Find the `ds_read_u16` instructions** in the trace
2. **Check latency** - high latency = conflicts/stalls
3. **Look at execution pattern** - are multiple reads serialized?
4. **Correlate with assembly** - map to lines 253-260 in your `.s` file

### Example Analysis:

```bash
# Decode trace to CSV
/opt/rocm-7.2.0/libexec/rocprofiler/att/att_to_csv.py \
  att_xor/ProductionTransposeKernel_*.att \
  -o xor_trace.csv

# Find LDS instructions
grep -i "ds_read" xor_trace.csv | head -20

# Look for high latency
awk -F',' '$4 > 100 {print}' xor_trace.csv  # Latency > 100 cycles
```

---

## Compare XOR vs No-XOR

Profile both versions and compare instruction latencies:

```bash
# Profile no-XOR version
rocprofv3 --att -o att_no_xor -- ./bin/aa_tutorial_14_01_row_major

# Profile XOR version
rocprofv3 --att -o att_xor -- ./bin/aa_tutorial_14_04_row_major_xor

# Compare the traces
diff <(grep ds_read att_no_xor/*.csv) <(grep ds_read att_xor/*.csv)
```

---

## Troubleshooting

### Problem: No .att files generated

**Check:**
```bash
# Verify ATT plugin is loaded
ls /opt/rocm-7.2.0/lib/rocprofiler/libatt_plugin.so

# Check output directory
ls -R att_xor/
```

**Solution:**
- Ensure kernel actually executed (check with `--hip-trace`)
- Try larger buffer: `--att-buffer-size 1024`
- Check GPU supports ATT (MI300X does)

### Problem: "ATT not supported on this GPU"

Your MI300X **does** support ATT. If you see this:
- Update ROCm to latest (you have 7.2.0 which is good)
- Check kernel permissions for GPU access

### Problem: Trace is truncated

**Solution:**
```bash
--att-buffer-size 1024  # Increase from default 256MB
```

### Problem: Too much data to analyze

**Focus on specific kernel:**
```bash
--kernel-include-regex "ProductionTranspose"
--att-consecutive-kernels 1  # Only first invocation
```

---

## Advanced: Perfetto Visualization Tips

### In Perfetto UI:

1. **Use WASD keys** to navigate timeline
2. **Click instruction** to see assembly in tooltip
3. **Select region** to see aggregated stats
4. **"m" key** to add markers
5. **Search** (Ctrl+F) for specific instructions like "ds_read"

### Custom Queries (SQL):

Perfetto supports SQL queries on trace data:
```sql
SELECT name, dur FROM slice WHERE name LIKE '%ds_read%'
```

---

## Expected Results for Your XOR Test

### What You Should See:

1. **8 `ds_read_u16` instructions** (lines 253-260 in assembly)
2. **Some reads have higher latency** than others
3. **Pattern of serialization** if bank conflicts occur
4. **XOR version should have lower average latency** than no-XOR

### Specific Things to Check:

**In the trace:**
- Do reads at offset:128 (line 257) show different behavior?
- Do reads at offset:256 (line 260) show higher latency?
- Are there gaps between consecutive reads (= stalls)?

**Assembly correlation:**
Match PC addresses in trace to lines in:
`/data0/aghamari/composable_kernel/04_row_major_xor-hip-amdgcn-amd-amdhsa-gfx942.s`

---

## Quick Reference Commands

```bash
# Basic ATT profiling
rocprofv3 --att -o results -- ./binary

# With LDS metrics
rocprofv3 --att -i metrics.txt -o results -- ./binary

# Specific CU/SIMD
rocprofv3 --att --att-target-cu 0 --att-simd-select 0xF -o results -- ./binary

# Decode to CSV
/opt/rocm-7.2.0/libexec/rocprofiler/att/att_to_csv.py trace.att -o output.csv

# View in Perfetto
# Open https://ui.perfetto.dev, load results_results.pftrace
```

---

## Next Steps

1. **Run ATT profiling** on your XOR kernel
2. **Load Perfetto trace** to see instruction timeline
3. **Decode to CSV** for detailed analysis
4. **Identify which `ds_read` instructions** cause conflicts
5. **Compare with assembly** to understand why

This will give you **exact instruction-level visibility** into where and why the LDS bank conflicts occur!

---

## Documentation Links

- ROCm Profiler: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
- Thread Trace: https://github.com/ROCm/rocprof-trace-decoder
- Perfetto UI: https://ui.perfetto.dev
- Your assembly: `/data0/aghamari/composable_kernel/04_row_major_xor-hip-amdgcn-amd-amdhsa-gfx942.s`
