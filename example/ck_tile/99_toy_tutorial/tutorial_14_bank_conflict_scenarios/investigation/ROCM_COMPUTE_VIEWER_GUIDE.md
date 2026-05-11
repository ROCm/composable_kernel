# ROCm Compute Viewer - Assembly-Level Thread Trace Visualization

## What You Have Generated

✅ **The directory to pass to ROCm Compute Viewer:**
```
ui_output_agent_17025_dispatch_1/
```

This directory contains JSON files that ROCm Compute Viewer uses to show you:
- Assembly code
- Instruction-level execution trace
- Exact timing of each instruction
- Which `ds_read_u16` instructions cause LDS bank conflicts

## Contents of Your ui_output Directory

```
ui_output_agent_17025_dispatch_1/
├── code.json          - Assembly code with instruction addresses
├── filenames.json     - Source file mapping
├── occupancy.json     - GPU occupancy data
└── realtime.json      - Timing/execution data
```

---

## Installing ROCm Compute Viewer

### Option 1: Download Pre-built Binary (Easiest)

1. **Download the latest release:**
   https://github.com/ROCm/rocprof-compute-viewer/releases

2. **For Linux, download:**
   `rocprof-compute-viewer-<version>-Linux-x86_64.tar.gz`

3. **Extract and run:**
   ```bash
   cd ~/Downloads
   tar -xzf rocprof-compute-viewer-*-Linux-x86_64.tar.gz
   cd rocprof-compute-viewer-*
   ./rocprof-compute-viewer
   ```

### Option 2: Build from Source

If pre-built binary doesn't work:

```bash
# Install dependencies
sudo apt install -y libgl1 qt6-base-dev qmake6 build-essential

# Clone and build
cd ~/
git clone https://github.com/ROCm/rocprof-compute-viewer.git
cd rocprof-compute-viewer
mkdir build && cd build
cmake .. -DQT_VERSION_MINOR=4
make -j$(nproc)

# Run
./rocprof-compute-viewer
```

---

## How to Use ROCm Compute Viewer

### Method 1: Command Line

```bash
# Pass the ui_output directory
./rocprof-compute-viewer /home/aghamari/composable_kernel/example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios/ui_output_agent_17025_dispatch_1
```

### Method 2: GUI (Easier)

1. **Launch the viewer:**
   ```bash
   ./rocprof-compute-viewer
   ```

2. **Import your trace:**
   - Click **Menu → Import → Rocprofv3 UI**
   - Or paste the path in the "Ui path" field:
     ```
     /home/aghamari/composable_kernel/example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios/ui_output_agent_17025_dispatch_1
     ```

3. **View the assembly trace!**

---

## What You'll See in the Viewer

The ROCm Compute Viewer will show:

### 1. **Assembly Code View**
- Every instruction in your kernel
- The exact `ds_read_u16` instructions from lines 253-260 of your assembly

### 2. **Execution Timeline**
- When each instruction executed
- Stalls/latency (shows conflicts!)
- Which threads/waves executed which instructions

### 3. **Instruction-Level Metrics**
- How many cycles each instruction took
- If there were stalls (= bank conflicts for LDS reads)
- Thread/wave mapping

### 4. **Occupancy**
- How many waves were active
- GPU utilization

---

## What to Look For - LDS Bank Conflicts

In the viewer, look for:

### 1. **Find the `ds_read_u16` Instructions**

You should see 8 consecutive reads:
```assembly
ds_read_u16 v14, v23
ds_read_u16 v15, v24
ds_read_u16 v16, v34
ds_read_u16 v17, v25
ds_read_u16 v18, v28 offset:128
ds_read_u16 v19, v26
ds_read_u16 v20, v27 offset:128
ds_read_u16 v21, v22 offset:256
```

### 2. **Check Execution Latency**

- **No conflicts:** All reads complete in ~4-10 cycles
- **With conflicts:** Some reads take 20-50+ cycles (serialization)

### 3. **Look for Gaps/Stalls**

In the timeline view:
- Smooth execution = no conflicts
- Gaps between instructions = stalls from bank conflicts

### 4. **Compare XOR vs No-XOR**

Profile both versions and compare in the viewer to see how XOR reduces conflicts!

---

## Complete Workflow

### Step 1: Generate Trace Data

```bash
cd /home/aghamari/composable_kernel/example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios

# Profile with ATT
rocprofv3 \
  --att \
  --att-library-path ~/rocm-tools/lib \
  --hip-trace \
  --kernel-trace \
  --output-format pftrace \
  -o xor_trace \
  -- ./04_row_major_xor_asm
```

This generates: `ui_output_agent_*_dispatch_*/`

### Step 2: Launch ROCm Compute Viewer

```bash
# Download viewer first (if not installed)
# Then launch with your trace directory
./rocprof-compute-viewer ui_output_agent_17025_dispatch_1/
```

### Step 3: Analyze in Viewer

- Navigate to assembly view
- Find `ds_read_u16` instructions
- Check timing/latency
- Identify which reads have conflicts

---

## Quick Reference

**Your trace directory:**
```
/home/aghamari/composable_kernel/example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios/ui_output_agent_17025_dispatch_1/
```

**To regenerate for different kernel:**
```bash
# Using convenience script
./profile_att.sh output_name ./your_binary

# Or manually with env variable
source ~/.bashrc
att-profile -o my_trace -- ./my_binary

# Then open the generated ui_output_* directory in viewer
```

**Download ROCm Compute Viewer:**
https://github.com/ROCm/rocprof-compute-viewer/releases

---

## Troubleshooting

### Problem: "No ui_output directory generated"

**Check:**
```bash
ls -la | grep ui_output
```

**Solution:**
Ensure you're using `--att-library-path ~/rocm-tools/lib` (the trace decoder is required)

### Problem: "Viewer doesn't open the directory"

**Check:**
```bash
ls ui_output_agent_*/
```

Ensure it has `.json` files inside.

### Problem: "Viewer crashes or shows empty"

**Try:**
- Use latest version of viewer
- Check if JSON files are valid: `cat ui_output_*/code.json`
- Re-run profiling with `-o new_output_name`

---

## Comparing XOR vs No-XOR

### Profile Both Versions:

```bash
# No-XOR version
att-profile -o trace_no_xor -- ./01_row_major

# XOR version
att-profile -o trace_xor -- ./04_row_major_xor_asm
```

### Open Both in Viewer:

1. Open trace_no_xor ui_output directory
2. Save screenshots/notes of `ds_read_u16` latencies
3. Open trace_xor ui_output directory
4. Compare latencies - XOR version should have lower latency!

---

## Expected Results

### What You Should See:

1. **Assembly code** matching your `.s` file (lines 253-260)
2. **8 `ds_read_u16` instructions** in sequence
3. **Variable latency** - some higher than others
4. **Stalls/gaps** where bank conflicts occur
5. **XOR version** should show lower average latency than no-XOR

This will give you **definitive proof** of where and why bank conflicts happen!

---

## Summary

✅ **You already have the correct file:** `ui_output_agent_17025_dispatch_1/`

✅ **Next step:** Download and run ROCm Compute Viewer

✅ **Pass the ui_output directory** to see assembly-level instruction trace

✅ **Identify exactly which `ds_read_u16` instructions** cause LDS bank conflicts

This is the tool you've been looking for! 🎯
