# CK Tile Unified Code Generators

Single source of truth for GEMM and Grouped Convolution kernel generation.

> **See also:** [Main Dispatcher README](../README.md) for installation and core concepts.

## Shared Infrastructure

Both GEMM and Grouped Conv generators share common code via `codegen_common.py`:
- `TileConfig` - Dataclass for tile dimensions
- `TraitConfigBase` - Base for kernel trait configurations with arch-aware validation
- `CommonTypeMappings` - Dtype-to-C++ type mappings
- `parallel_generate()` - Parallel kernel generation with per-kernel progress logging
- Arch-aware expansion helpers (`valid_wave_configs`, `valid_warp_configs`, etc.)

## Quick Start

### GEMM

```bash
cd dispatcher/codegen

# Generate standard FP16 kernels
python3 unified_gemm_codegen.py \
    --output-dir ../build/generated_kernels \
    --datatype fp16 \
    --layout rcr \
    --variants standard

# Generate all variants
python3 unified_gemm_codegen.py \
    --output-dir ../build/generated_kernels \
    --variants standard preshuffle multi_d
```

### Grouped Convolution

```bash
cd dispatcher/codegen

# Generate forward FP16 grouped conv kernels
python3 unified_grouped_conv_codegen.py \
    --output-dir ../build/generated_kernels \
    --datatype fp16 \
    --variant forward \
    --ndim-spatial 2

# Generate backward data kernels
python3 unified_grouped_conv_codegen.py \
    --output-dir ../build/generated_kernels \
    --variant backward_data \
    --ndim-spatial 2
```

## Using from Python

```python
from ctypes_utils import CodegenRunner, KernelConfig

# Generate from specific config
config = KernelConfig(tile_m=256, tile_n=256, tile_k=64)
codegen = CodegenRunner()
result = codegen.generate_from_config(config)

# Generate variant
result = codegen.generate("preshuffle")

# Generate all
results = codegen.generate_all()
```

## Command Line Options

| Option | Values | Description |
|--------|--------|-------------|
| `--output-dir` | path | Output directory |
| `--datatype` | `fp16`, `bf16`, `fp32`, `int8` | Data type |
| `--layout` | `rcr`, `rrr`, `crr`, `ccr` | Matrix layouts |
| `--gpu-target` | `gfx942`, `gfx90a`, `gfx950` | Target GPU |
| `--variants` | `standard`, `preshuffle`, `multi_d` | Kernel variants |
| `--preselected` | `fp16_rcr_essential`, etc. | Predefined kernel set |

### Layout Notation

- `R` = Row-major, `C` = Column-major
- Order: A, B, C (e.g., `rcr` = A row, B col, C row)

## Variants

### Standard
Basic GEMM: `C = A x B`

### PreShuffle
Optimized weight access with LDS pre-shuffling. Best for large matrices.

### Multi-D
Element-wise fusion: `C = op(A x B + D0 + D1 + ...)`

Supported ops: `PassThrough`, `MultiDAdd`, `Relu`, `Gelu`, `Sigmoid`, `Tanh`

## Output Structure

```
generated_kernels/
|---- gemm_fp16_rcr_compv4_..._128x128x32_....hpp          # GEMM kernels
|---- gemm_fp16_rcr_compv4_..._preshuffle.hpp
|---- gemm_fp16_rcr_compv4_..._multid_Relu_d1.hpp
|---- grouped_conv_fwd_fp16_nhwgc_..._128x128x32_....hpp   # Grouped conv kernels
+---- ...
```

## Configuration Files

### arch_specs.json

GPU architecture specifications (single source of truth):

```json
{
  "architectures": {
    "gfx942": {
      "family": "cdna3",
      "warp_size": 64,
      "warp_configs": [[2, 2, 1], [4, 4, 1]],
      ...
    }
  }
}
```

### preselected_kernels.py

Curated kernel sets for common use cases.

## Adding New GPU Support

See [ADDING_NEW_GPU.md](ADDING_NEW_GPU.md) for complete guide.

Quick steps:
1. Edit `arch_specs.json`
2. Run `python generate_arch_specs.py`
3. Rebuild

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Arguments not supported" | Check tile config validity |
| Missing element-wise op | Check `elementwise_ops.hpp` |
| Compilation errors | Verify C++17, include paths |

---

> **More info:** See [../README.md](../README.md) for full documentation.
