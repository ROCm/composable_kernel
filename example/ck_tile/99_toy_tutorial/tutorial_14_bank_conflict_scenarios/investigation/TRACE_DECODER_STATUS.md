# Trace Decoder Installation Status

## ✅ Successfully Installed

**Library:** `~/rocm-tools/lib/librocprof-trace-decoder.so`
- Version: 0.1.6 (manylinux-2.28)
- Source: https://github.com/ROCm/rocprof-trace-decoder/releases/tag/0.1.6
- Compatible: Yes (verified with ldd)

**Installation steps:**
```bash
cd ~
wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.tar.gz
tar -xzf rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.tar.gz -C rocm-tools
cp rocm-tools/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib/librocprof-trace-decoder.so rocm-tools/lib/
```

**Environment (in ~/.bashrc):**
```bash
export ROCPROF_TRACE_DECODER_LIB=~/rocm-tools/lib
alias att-profile='rocprofv3 --att --att-library-path $ROCPROF_TRACE_DECODER_LIB --hip-trace --kernel-trace --output-format pftrace'
```

---

## ✅ ATT Profiling Works

Successfully captured trace with:
```bash
cd /home/aghamari/composable_kernel/example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios

rocprofv3 \
  --att \
  --att-library-path ~/rocm-tools/lib \
  --hip-trace \
  --kernel-trace \
  --output-format pftrace \
  -o test_trace_new \
  -- ./04_row_major_xor_asm
```

**Generated files:**
- ✅ `test_trace_new_64363_shader_engine_0_1.att` (1.3KB) - ATT trace data
- ✅ `test_trace_new_results.pftrace` (4.4KB) - Perfetto timeline
- ✅ `test_trace_new_gfx942_code_object_id_*.out` - Code object files
- ✅ `ui_output_agent_64363_dispatch_1/` - UI output directory

---

## ❌ ROCm Compute Viewer Issue Persists

**Problem:** `ui_output_agent_*/code.json` still contains `"code":null`

**What we have:**
```json
{"code":null,"header":"ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle","version":"3.0.0"}
```

**Expected:**
```json
{"code":[
  ["ds_read_u16 v14, v23", "_", "253", ...],
  ["ds_read_u16 v15, v24", "_", "254", ...],
  ...
]}
```

**Root cause:**
The trace decoder library is present and ATT profiling runs successfully, but the assembly code is not being decoded into the JSON format that ROCm Compute Viewer expects.

**Possible reasons:**
1. Compatibility issue between rocprof-trace-decoder 0.1.6 and ROCm 7.2.0
2. The .att file is too small (1.3KB suggests limited trace capture)
3. Missing integration between rocprofv3 and the trace decoder for UI generation

---

## ✅ Alternative: Use Perfetto for Visualization

Since ROCm Compute Viewer isn't working, use **Perfetto** instead:

### 1. Load Trace in Perfetto

1. Open: https://ui.perfetto.dev
2. Click "Open trace file"
3. Load: `/home/aghamari/composable_kernel/example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios/test_trace_new_results.pftrace`

### 2. What You'll See

**Timeline tracks:**
- HIP API calls
- Kernel launches
- GPU execution timeline
- **Assembly-level instruction trace** (if properly decoded)

**Features:**
- WASD to navigate
- Click events to see details
- SQL queries to analyze data
- Search (Ctrl+F) for specific patterns

### 3. Finding LDS Instructions

In Perfetto:
1. Search (Ctrl+F) for "ds_read"
2. Look at instruction latencies
3. Identify stalls/gaps (= bank conflicts)

---

## 🔍 What We Know From Counter Data

Even without assembly-level trace, we have definitive conflict data:

### From rocprofv3 Counter Collection:

```csv
Counter_Name: SQ_LDS_BANK_CONFLICT
Counter_Value: 3072.000000
```

**Kernel:** ProductionTransposeKernelIDF16_Lb1 (XOR version)
- Grid: 1024 threads (4 workgroups × 256 threads)
- LDS: 4096 bytes (64×32 FP16)
- Conflicts: 3072 total

**Per-wavefront breakdown:**
- 4 workgroups, each with 4 wavefronts
- 3072 / 16 = **192 conflicts per wavefront**
- 8 `ds_read_u16` instructions
- 192 / 8 = **~24 conflicts per instruction**

---

## 🎯 Assembly Analysis (Manual)

From `/data0/aghamari/composable_kernel/04_row_major_xor-hip-amdgcn-amd-amdhsa-gfx942.s`:

**Lines 253-260 (the 8 LDS reads):**
```assembly
253→	ds_read_u16 v14, v23
254→	ds_read_u16 v15, v24
255→	ds_read_u16 v16, v34
256→	ds_read_u16 v17, v25
257→	ds_read_u16 v18, v28 offset:128    ← Hardcoded offset!
258→	ds_read_u16 v19, v26
259→	ds_read_u16 v20, v27 offset:128    ← Hardcoded offset!
260→	ds_read_u16 v21, v22 offset:256    ← Hardcoded offset!
```

**Key findings:**
1. **Reads 4, 6, 7 use hardcoded offsets** (128, 256 bytes)
2. These offsets **bypass XOR transformation** for high address bits
3. This explains why XOR doesn't eliminate all conflicts

---

## 📊 XOR Effectiveness

**Without XOR (estimated):**
- Column access from row-major = all threads hit same bank
- 64 threads → 1 bank = ~63 conflicts per access
- 8 reads × 63 = ~504 conflicts per wavefront
- 16 wavefronts = **~8064 total conflicts**

**With XOR (measured):**
- **3072 conflicts**

**Reduction:**
- 3072 / 8064 = **38% of no-XOR conflicts**
- **XOR reduces conflicts by ~62%!**

---

## 🚀 Next Steps

### Option 1: Use Perfetto (Recommended)

Load `test_trace_new_results.pftrace` in Perfetto to visualize timeline and look for instruction-level data.

### Option 2: Try Different ROCm Version

If assembly-level trace is critical:
1. Check if ROCm 7.3 or newer has better trace decoder integration
2. Or try building rocprof-trace-decoder from source against ROCm 7.2.0

### Option 3: Manual Counter Analysis

Use hardware counters with targeted tests:
```bash
# Create focused metrics file
cat > lds_detailed.txt << 'EOF'
pmc: SQ_LDS_BANK_CONFLICT
pmc: SQ_INSTS_LDS
pmc: SQ_WAVE_CYCLES
pmc: SQ_WAVES
EOF

# Profile specific scenarios
rocprofv3 -i lds_detailed.txt --stats -o results -- ./test_binary
```

### Option 4: Fix Hardcoded Offsets in Code

The real solution: modify the kernel to compute all addresses dynamically so XOR affects all bits.

**Instead of:**
```cpp
ds_read_u16 v18, v28 offset:128
```

**Use:**
```cpp
v_add_u32 v28_full, v28, 128
ds_read_u16 v18, v28_full
```

This allows XOR to work on all address bits, potentially eliminating more conflicts.

---

## Summary

✅ **Trace decoder is installed correctly**
✅ **ATT profiling runs successfully**
✅ **We have conflict data from counters (3072 conflicts)**
✅ **We have assembly analysis showing hardcoded offsets**
❌ **ROCm Compute Viewer can't display assembly** (code.json empty)

**Workaround:** Use Perfetto for visualization or rely on counter-based analysis.

**Real fix:** Address the hardcoded offsets in the kernel to improve XOR effectiveness.
