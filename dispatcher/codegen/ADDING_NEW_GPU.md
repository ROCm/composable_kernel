# Adding New GPU Architecture Support

Guide for adding support for a new AMD GPU architecture to the CK Tile Dispatcher.

> **See also:** [Main Dispatcher README](../README.md) | [Codegen README](README.md)

## Overview

The dispatcher uses `arch_specs.json` as the **single source of truth** for GPU specifications:

```
arch_specs.json -> generate_arch_specs.py -> arch_specs_generated.py (Python)
                                        -> arch_specs_generated.hpp (C++)
```

## Quick Start

```bash
# 1. Edit arch_specs.json
# 2. Run generator
python generate_arch_specs.py
# 3. Rebuild
cd ../build && cmake --build . -j8
# 4. Test
ctest
```

## Step-by-Step Guide

### Step 1: Edit arch_specs.json

Add new architecture under `"architectures"`:

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

### Step 2: Configuration Fields

| Field | Description | Example |
|-------|-------------|---------|
| `family` | GPU family | `"cdna3"`, `"rdna4"` |
| `description` | Human-readable name | `"AMD Instinct MI300"` |
| `warp_size` | Wave/warp size | `64` (CDNA), `32` (RDNA) |
| `lds_capacity_kb` | LDS memory in KB | `64` |
| `warp_configs` | Valid `[warp_m, warp_n, warp_k]` | `[[2,2,1], [4,4,1]]` |
| `warp_tile_combos` | Warp tiles per dtype | See below |

### Step 3: Warp Tile Combinations

Map data type combinations to valid warp tile sizes:

```json
"warp_tile_combos": {
  "fp16_fp16_fp16": [[32, 32, 8], [16, 16, 16], [32, 32, 16]],
  "bf16_bf16_bf16": [[32, 32, 8], [16, 16, 16]],
  "fp8_fp8_fp16": [[32, 32, 16], [32, 32, 32]],
  "int8_int8_int32": [[16, 16, 32], [32, 32, 16]]
}
```

Key format: `{A_dtype}_{B_dtype}_{C_dtype}`

### Step 4: Run Generator

```bash
cd dispatcher/codegen
python generate_arch_specs.py
```

This generates:
- `arch_specs_generated.py` (Python module)
- `../include/ck_tile/dispatcher/arch_specs_generated.hpp` (C++ header)

### Step 5: Rebuild and Test

```bash
cd ../build
cmake --build . -j8
ctest --output-on-failure
```

### Step 6: Verify

```python
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

## Reference

### Supported Data Types

| Key | Description |
|-----|-------------|
| `fp16` | Half precision (16-bit) |
| `bf16` | Brain float 16 |
| `fp32` | Single precision (32-bit) |
| `fp64` | Double precision (64-bit) |
| `fp8` | 8-bit float (E4M3) |
| `bf8` | 8-bit brain float (E5M2) |
| `int8` | 8-bit integer |
| `int4` | 4-bit integer |

### GPU Families

| Family | Description |
|--------|-------------|
| `cdna2` | MI200 series (gfx90a) |
| `cdna3` | MI300 series (gfx942) |
| `cdna4` | MI350 series (gfx950) |
| `rdna3` | RX 7000 series (gfx1100) |
| `rdna4` | RX 9000 series (gfx1201) |

### Pipeline LDS Limits

| Pipeline | LDS Limit |
|----------|-----------|
| `compv4` | 32 KB |
| `preshufflev2` | 32 KB |
| `default` | 64 KB |

## Troubleshooting

### "Unknown GPU architecture"

1. Check architecture key matches exactly (e.g., `"gfx942"` not `"GFX942"`)
2. Verify you ran `generate_arch_specs.py`
3. Rebuild C++ code

### Kernels being rejected

```python
from arch_filter import ArchFilter, KernelConfig

filter = ArchFilter("gfx942")
result = filter.validate_kernel(config)
print(f"Valid: {result.valid}")
for error in result.errors:
    print(f"  Error: {error}")
```

### Missing warp tile combination

1. Check `warp_tile_combos` in `arch_specs.json`
2. Ensure `[warp_tile_m, warp_tile_n, warp_tile_k]` is in the list
3. Verify data type key format

## File Structure

```
codegen/
|---- arch_specs.json              # Single source of truth (EDIT THIS)
|---- generate_arch_specs.py       # Generator script
|---- arch_specs_generated.py      # Generated Python module
+---- ADDING_NEW_GPU.md           # This file

include/ck_tile/dispatcher/
|---- arch_specs_generated.hpp     # Generated C++ header
+---- arch_filter.hpp              # C++ filter
```

## Best Practices

1. **Test thoroughly** - Run all tests after adding a new GPU
2. **Start minimal** - Add only validated configurations
3. **Document sources** - Note where warp tile combinations came from
4. **Keep in sync** - If using tile_engine, keep both updated

---

> **More info:** See [../README.md](../README.md) for full documentation.
