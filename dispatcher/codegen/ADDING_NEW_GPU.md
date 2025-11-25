# Adding New GPU Architecture Support

This guide explains how to add support for a new AMD GPU architecture to the CK Tile Dispatcher.

## Overview

The dispatcher uses a **single source of truth** (`arch_specs.json`) for all GPU architecture specifications. This file is used to generate both Python and C++ code, ensuring consistency across the codebase.

```
arch_specs.json  ──►  generate_arch_specs.py  ──►  arch_specs_generated.py (Python)
                                              ──►  arch_specs_generated.hpp (C++)
```

## Quick Start

To add support for a new GPU (e.g., `gfx1100`):

1. **Edit `arch_specs.json`** - Add the new architecture entry
2. **Run the generator** - `python generate_arch_specs.py`
3. **Rebuild** - `cmake --build . -j8`
4. **Test** - Run tests with `ctest`

## Step-by-Step Guide

### Step 1: Edit arch_specs.json

Open `dispatcher/codegen/arch_specs.json` and add a new entry under `"architectures"`:

```json
{
  "architectures": {
    "gfx1100": {
      "family": "rdna3",
      "description": "AMD Radeon RX 7000 series (RDNA3)",
      "warp_size": 32,
      "lds_capacity_kb": 64,
      "warp_configs": [
        [2, 4, 1],
        [1, 8, 1],
        [8, 1, 1],
        [4, 2, 1]
      ],
      "warp_tile_combos": {
        "fp16_fp16_fp16": [[16, 16, 16], [32, 32, 16]],
        "bf16_bf16_bf16": [[16, 16, 16], [32, 32, 16]]
      }
    }
  }
}
```

### Step 2: Understand the Configuration Fields

| Field | Description | Example |
|-------|-------------|---------|
| `family` | GPU family identifier | `"cdna3"`, `"rdna4"` |
| `description` | Human-readable description | `"AMD Instinct MI300 series"` |
| `warp_size` | Wave/warp size | `64` for CDNA, `32` for RDNA |
| `lds_capacity_kb` | LDS memory capacity in KB | `64` |
| `warp_configs` | Valid `[warp_m, warp_n, warp_k]` combinations | `[[1,4,1], [2,2,1]]` |
| `warp_tile_combos` | Valid warp tile sizes per data type | See below |

### Step 3: Determine Warp Tile Combinations

The `warp_tile_combos` field maps data type combinations to valid warp tile configurations:

```json
"warp_tile_combos": {
  "fp16_fp16_fp16": [[32, 32, 8], [16, 16, 16], [32, 32, 16]],
  "bf16_bf16_bf16": [[32, 32, 8], [16, 16, 16]],
  "fp8_fp8_fp16": [[32, 32, 16], [32, 32, 32]],
  "int8_int8_int32": [[16, 16, 32], [32, 32, 16]]
}
```

The key format is `{A_dtype}_{B_dtype}_{C_dtype}` where:
- `A_dtype`: Input matrix A data type
- `B_dtype`: Input matrix B data type
- `C_dtype`: Output matrix C data type

### Step 4: Run the Generator

```bash
cd dispatcher/codegen
python generate_arch_specs.py
```

This generates:
- `arch_specs_generated.py` - Python module
- `include/ck_tile/dispatcher/arch_specs_generated.hpp` - C++ header

### Step 5: Rebuild and Test

```bash
cd dispatcher/build
cmake --build . -j8
ctest --output-on-failure
```

### Step 6: Verify with the Filter

Test your new architecture:

```python
# Python
from arch_filter import ArchFilter

filter = ArchFilter("gfx1100")
is_valid = filter.is_kernel_valid(
    datatype_a="fp16", datatype_b="fp16", datatype_c="fp16",
    tile_m=128, tile_n=128, tile_k=32,
    warp_m=2, warp_n=2, warp_k=1,
    warp_tile_m=16, warp_tile_n=16, warp_tile_k=16
)
print(f"Valid: {is_valid}")
```

```cpp
// C++
#include "ck_tile/dispatcher/arch_filter.hpp"

ArchFilter filter("gfx1100");
bool valid = filter.is_valid(kernel_key);
```

## Configuration Reference

### Supported Data Types

| Key | Description |
|-----|-------------|
| `fp16` | Half precision (16-bit float) |
| `bf16` | Brain float 16 |
| `fp32` | Single precision (32-bit float) |
| `fp8` | 8-bit float (E4M3) |
| `bf8` | 8-bit brain float (E5M2) |
| `int8` | 8-bit integer |
| `int32` | 32-bit integer |

### GPU Families

| Family | Description |
|--------|-------------|
| `cdna2` | MI200 series (gfx90a) |
| `cdna3` | MI300 series (gfx942) |
| `cdna4` | MI350 series (gfx950) |
| `rdna3` | RX 7000 series (gfx1100, gfx1101, gfx1102) |
| `rdna4` | RX 9000 series (gfx1201) |

### Pipeline LDS Limits

Different pipeline types have different LDS capacity limits:

| Pipeline | LDS Limit |
|----------|-----------|
| `compv4` | 32 KB |
| `preshufflev2` | 32 KB |
| `default` | 64 KB |

### Unsupported Trait Combinations

Some pipeline/epilogue/scheduler combinations don't work together. These are defined in `unsupported_trait_combos`:

```json
"unsupported_trait_combos": {
  "combinations": [
    ["compv3", "cshuffle", "interwave"],
    ["compv4", "cshuffle", "interwave"]
  ]
}
```

## Troubleshooting

### "Unknown GPU architecture" error

Make sure:
1. The architecture key matches exactly (e.g., `"gfx942"`, not `"GFX942"`)
2. You ran `generate_arch_specs.py` after editing `arch_specs.json`
3. You rebuilt the C++ code

### Kernels being rejected

Check validation errors:

```python
from arch_filter import ArchFilter, KernelConfig

filter = ArchFilter("gfx942")
config = KernelConfig(
    datatype_a="fp16", datatype_b="fp16", datatype_c="fp16",
    tile_m=256, tile_n=256, tile_k=64,
    warp_m=2, warp_n=2, warp_k=1,
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16
)
result = filter.validate_kernel(config)
print(f"Valid: {result.valid}")
for error in result.errors:
    print(f"  Error: {error}")
for warning in result.warnings:
    print(f"  Warning: {warning}")
```

### Missing warp tile combination

If you get "Invalid warp tile" errors:
1. Check `warp_tile_combos` in `arch_specs.json` for your architecture
2. Ensure the combination `[warp_tile_m, warp_tile_n, warp_tile_k]` is in the list
3. Verify the data type key (e.g., `fp16_fp16_fp16`)

## File Structure

```
dispatcher/
├── codegen/
│   ├── arch_specs.json              # Single source of truth (EDIT THIS)
│   ├── generate_arch_specs.py       # Generator script
│   ├── arch_specs_generated.py      # Generated Python module
│   ├── arch_filter.py               # Python filter (uses generated module)
│   └── ADDING_NEW_GPU.md           # This file
│
└── include/ck_tile/dispatcher/
    ├── arch_specs_generated.hpp     # Generated C++ header
    └── arch_filter.hpp              # C++ filter (uses generated header)
```

## Best Practices

1. **Test thoroughly** - Run all tests after adding a new GPU
2. **Start minimal** - Add only the configurations you've validated
3. **Document sources** - Note where you got the warp tile combinations from
4. **Update tile_engine** - If using both systems, keep them in sync

