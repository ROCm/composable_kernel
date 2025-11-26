# CK Tile GEMM Unified Code Generator

Single source of truth for all GEMM kernel generation.

> **See also:** [Main Dispatcher README](../README.md) for installation and core concepts.

## Quick Start

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
Basic GEMM: `C = A × B`

### PreShuffle
Optimized weight access with LDS pre-shuffling. Best for large matrices.

### Multi-D
Element-wise fusion: `C = op(A × B + D0 + D1 + ...)`

Supported ops: `PassThrough`, `MultiDAdd`, `Relu`, `Gelu`, `Sigmoid`, `Tanh`

## Output Structure

```
generated_kernels/
├── gemm_fp16_rcr_compv4_..._128x128x32_....hpp
├── gemm_fp16_rcr_compv4_..._preshuffle.hpp
├── gemm_fp16_rcr_compv4_..._multid_Relu_d1.hpp
└── ...
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
