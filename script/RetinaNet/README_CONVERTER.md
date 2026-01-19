# PyTorch to CK Profiler Conversion Tools

This directory contains two Python scripts to convert PyTorch convolution operations to CK Profiler format and execute them.

## Overview

1. **`convert_pytorch_to_ck.py`** - Converts PyTorch JSON to CK Profiler configuration JSON
2. **`run_ck_profiler.py`** - Executes CK profilers for each configuration

## Workflow

```bash
# Step 1: Convert PyTorch JSON to CK Profiler JSON
python convert_pytorch_to_ck.py build/test-data/conv_repros_ir.json ck_profiler_configs.json

# Step 2: Execute profilers
python run_ck_profiler.py ck_profiler_configs.json --profiler-path ./build/bin
```

---

## Script 1: convert_pytorch_to_ck.py

### Description
Converts PyTorch convolution operations from `conv_repros_ir.json` to CK Profiler configuration format.

### Supported Operations
- `aten::miopen_convolution` → Forward convolution
- `aten::convolution_backward` → Backward data and/or backward weight

### Configuration
The script uses these hardcoded settings:
- **Data type**: FP16 (data_type=1)
- **Layout**: NGCHW_GKCYX_NGKHW (layout=3)
- **Index type**: 32-bit (index_type=0, for forward only)
- **Verification**: Disabled (verify=0)
- **Initialization**: Integer values (init=1)
- **Logging**: Disabled (log=0)
- **Timing**: Enabled (time=1)
- **Split-K**: "all" (for backward operations)

### Usage

```bash
# Basic usage (uses defaults)
python convert_pytorch_to_ck.py

# Specify input and output files
python convert_pytorch_to_ck.py input.json output.json

# Help
python convert_pytorch_to_ck.py --help
```

### Default Files
- **Input**: `build/test-data/conv_repros_ir.json`
- **Output**: `ck_profiler_configs.json`

### Output Format

The output JSON contains an array of configurations:

```json
[
  {
    "operation_type": "profile_grouped_conv_fwd",
    "profiler_args": {
      "data_type": 1,
      "layout": 3,
      "index_type": 0,
      "verify": 0,
      "init": 1,
      "log": 0,
      "time": 1,
      "num_dim_spatial": 2,
      "G": 32,
      "N": 32,
      "K": 4,
      "C": 4,
      "filter_spatial": [3, 3],
      "input_spatial": [200, 200],
      "strides": [1, 1],
      "dilations": [1, 1],
      "left_pads": [1, 1],
      "right_pads": [1, 1]
    },
    "metadata": {
      "priority_rank": 1,
      "pytorch_op": "aten::miopen_convolution",
      "description": "Forward conv: G=32, N=32, K=4, C=4, [3, 3] kernel, [200, 200] input"
    }
  }
]
```

### Example Output

```
Converting PyTorch convolutions to CK Profiler format...

✓ [1/78] Converted: aten::miopen_convolution (rank 1) → fwd
✓ [2/78] Converted: aten::convolution_backward (rank 2) → bwd_data
✓ [2/78] Converted: aten::convolution_backward (rank 2) → bwd_weight
...

============================================================
Conversion Summary
============================================================
Total PyTorch operations:     78
  ✓ Forward convolutions:       30
  ✓ Backward data convolutions: 35
  ✓ Backward weight convolutions: 35
  ⚠ Skipped operations:         0

Total CK profiler configs:    100
Output file:                  ck_profiler_configs.json
============================================================

Conversion completed successfully!
```

---

## Script 2: run_ck_profiler.py

### Description
Executes CK Profiler binaries for each configuration in the JSON file, with support for dry-run, filtering, and result collection.

### Usage

```bash
# Run all configurations
python run_ck_profiler.py ck_profiler_configs.json

# Run first 10 only (for testing)
python run_ck_profiler.py ck_profiler_configs.json --max-ops 10

# Dry run (show commands without executing)
python run_ck_profiler.py ck_profiler_configs.json --dry-run

# Specify profiler path
python run_ck_profiler.py ck_profiler_configs.json --profiler-path ./build/bin

# Quiet mode (less verbose output)
python run_ck_profiler.py ck_profiler_configs.json --quiet

# Don't save results
python run_ck_profiler.py ck_profiler_configs.json --no-save

# Custom results output file
python run_ck_profiler.py ck_profiler_configs.json --output my_results.json
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `config_file` | CK Profiler configuration JSON file | (required) |
| `--profiler-path` | Path to CK profiler binaries | `./build/bin` |
| `--max-ops` | Maximum number of operations to execute | All |
| `--dry-run` | Show commands without executing | False |
| `--quiet` | Reduce output verbosity | False |
| `--no-save` | Do not save results to JSON | False |
| `--output` | Output file for results | `profiler_results.json` |

### Example Output

```
Executing CK Profiler for 100 configurations...

======================================================================
[1/100] Forward conv: G=32, N=32, K=4, C=4, [3, 3] kernel, [200, 200] input
Priority Rank: 1
PyTorch Op:    aten::miopen_convolution
======================================================================
Command: ./build/bin/profile_grouped_conv_fwd 1 3 0 0 1 0 1 2 32 32 4 4 3 3 200 200 1 1 1 1 1 1 1 1

✓ SUCCESS (completed in 2.34s)

Output:
...profiler output...

...

======================================================================
Execution Summary
======================================================================
Total configurations: 100
  ✓ Successful:  95
  ✗ Failed:      5

Success rate: 95.0%
======================================================================

Results saved to: profiler_results.json
```

### Results File Format

The `profiler_results.json` file contains:

```json
{
  "timestamp": "2026-01-19T05:48:00",
  "stats": {
    "total": 100,
    "success": 95,
    "failed": 5,
    "skipped": 0
  },
  "results": [
    {
      "operation": "profile_grouped_conv_fwd",
      "description": "Forward conv: G=32, N=32, K=4, C=4, [3, 3] kernel, [200, 200] input",
      "priority_rank": 1,
      "success": true,
      "returncode": 0,
      "elapsed_time": 2.34,
      "command": "./build/bin/profile_grouped_conv_fwd ...",
      "error": null,
      "stdout": "...truncated..."
    }
  ]
}
```

---

## Requirements

- Python 3.6+
- CK profiler binaries built in `./build/bin` (or specify with `--profiler-path`)
- Input JSON file with PyTorch convolution operations

## Notes

### Layout Mapping
PyTorch uses **NCHW** layout, which maps to CK's **NGCHW_GKCYX_NGKHW** (layout=3):
- Input: [N, G, C, Hi, Wi]
- Weight: [G, K, C, Y, X]
- Output: [N, G, K, Ho, Wo]

### Per-Group Channels
For grouped convolutions:
- `C_per_group = C_total / groups`
- `K_per_group = K_total / groups`

The converter automatically computes these values.

### Backward Operations
The `aten::convolution_backward` operation can compute:
- **Backward data** (gradient w.r.t. input) when `output_mask[0]=true`
- **Backward weight** (gradient w.r.t. weight) when `output_mask[1]=true`
- Both if both flags are true

### Split-K Parameter
For backward operations, split_k is set to "all" which instructs the profiler to test all split-K values: -1, 1, 2, 4, 8, 16, 32, 64, 128.

---

## Troubleshooting

### "Profiler executable not found"
Ensure CK profilers are built:
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make profile_grouped_conv_fwd profile_grouped_conv_bwd_data profile_grouped_conv_bwd_weight
```

### "Input file not found"
Check that the PyTorch JSON file exists:
```bash
ls -l build/test-data/conv_repros_ir.json
```

### Conversion warnings
Yellow warnings indicate operations that couldn't be converted (e.g., non-convolution operations). These are expected and don't indicate errors.

---

