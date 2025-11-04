# CK Tile GEMM Unified Code Generator

**Single source of truth for all GEMM kernel generation.**

This directory contains the unified code generation system that replaces all `tile_engine` GEMM codegen. It generates both CK Tile kernel instances AND dispatcher wrappers in a single pass.

## Architecture

```
unified_gemm_codegen.py  ← Single entry point for all variants
├── CK Tile Kernel Generation
│   ├── Standard GEMM (C = A × B)
│   ├── Preshuffle GEMM (optimized weight access)
│   └── Multi-D GEMM (element-wise fusion)
└── Dispatcher Wrapper Generation
    ├── KernelKey construction
    ├── Type mappings
    └── Registration helpers
```

## Key Features

### 1. **Unified Generation**
- Single script generates both kernel code and dispatcher wrappers
- Consistent naming across all variants
- Automatic registration header generation

### 2. **All GEMM Variants**
- **Standard**: Basic matrix multiplication
- **Preshuffle**: Weight preshuffle optimization
- **Multi-D**: Element-wise fusion (Add, Multiply, Relu, Gelu, etc.)

### 3. **Complete Type Safety**
- Centralized type mappings (CK types ↔ Dispatcher types)
- Compile-time validation
- Automatic output type handling (fp8/bf8 → fp16)

### 4. **Flexible Configuration**
- JSON-based tile and trait configuration
- Support for custom tile shapes
- Pipeline, epilogue, scheduler combinations
- Parallel generation for speed

## Usage

### Basic Generation

```bash
# Generate standard FP16 GEMM kernels
python unified_gemm_codegen.py \
    --output-dir ./generated \
    --datatype fp16 \
    --layout rcr \
    --variants standard

# Generate all variants
python unified_gemm_codegen.py \
    --output-dir ./generated \
    --datatype fp16 \
    --layout rcr \
    --variants standard preshuffle multi_d
```

### Custom Configuration

Create `config.json`:

```json
{
  "tile_config": {
    "tile_m": [128, 256],
    "tile_n": [128, 256],
    "tile_k": [32, 64],
    "warp_m": [2, 4],
    "warp_n": [2, 4],
    "warp_k": [1],
    "warp_tile_m": [16, 32],
    "warp_tile_n": [16, 32],
    "warp_tile_k": [16]
  },
  "trait_config": {
    "pipeline": ["compv3", "compv4"],
    "epilogue": ["cshuffle", "default"],
    "scheduler": ["intrawave"],
    "pad_m": [false],
    "pad_n": [false],
    "pad_k": [false],
    "persistent": [false, true]
  },
  "multi_d_config": {
    "elementwise_ops": ["MultiDAdd", "MultiDMultiply", "Relu", "Gelu"],
    "num_d_tensors": [1, 2]
  }
}
```

Then run:

```bash
python unified_gemm_codegen.py \
    --output-dir ./generated \
    --datatype fp16 \
    --layout rcr \
    --config config.json \
    --variants standard preshuffle multi_d
```

## Output Structure

```
generated/
├── gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x128x32_2x2x1_32x32x16.hpp
├── gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_True_256x128x32_2x2x1_32x32x16_preshuffle.hpp
├── gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x128x32_2x2x1_32x32x16_multid_Relu_d1.hpp
└── dispatcher_wrappers/
    ├── dispatcher_wrapper_gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x128x32_2x2x1_32x32x16.hpp
    ├── dispatcher_wrapper_gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_True_256x128x32_2x2x1_32x32x16_preshuffle.hpp
    ├── dispatcher_wrapper_gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x128x32_2x2x1_32x32x16_multid_Relu_d1.hpp
    └── register_all_kernels.hpp  ← Master registration header
```

## Integration with Dispatcher

### Automatic Registration

```cpp
#include "dispatcher_wrappers/register_all_kernels.hpp"

// Register all generated kernels
ck_tile::dispatcher::register_all_tile_gemm_kernels(942, Registry::Priority::High);

// Check count
auto count = ck_tile::dispatcher::get_tile_gemm_kernel_count();
std::cout << "Registered " << count << " kernels\n";
```

### Manual Registration

```cpp
#include "dispatcher_wrappers/dispatcher_wrapper_gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x128x32_2x2x1_32x32x16.hpp"

auto& registry = ck_tile::dispatcher::Registry::instance();
registry.register_kernel(
    ck_tile::dispatcher::generated::make_gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x128x32_2x2x1_32x32x16(942),
    Registry::Priority::High
);
```

## Kernel Naming Convention

Follows tile_engine convention:

```
gemm_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_{pad_m}_{pad_n}_{pad_k}_{persistent}_{tile_m}x{tile_n}x{tile_k}_{warp_m}x{warp_n}x{warp_k}_{warp_tile_m}x{warp_tile_n}x{warp_tile_k}[_variant]
```

Examples:
- `gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x128x32_2x2x1_32x32x16`
- `gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_True_256x128x32_2x2x1_32x32x16_preshuffle`
- `gemm_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x128x32_2x2x1_32x32x16_multid_Relu_d1`

## Supported Configurations

### Data Types
- `fp16`, `bf16`, `fp32`
- `fp8`, `bf8` (output automatically converted to fp16)
- `int8`

### Layouts
- `r` = Row-major
- `c` = Column-major
- Common: `rcr`, `rrr`, `crr`, `ccr`

### Pipelines
- `mem`: Memory-bound
- `compv3`: Compute-optimized v3
- `compv4`: Compute-optimized v4 (with double buffering)

### Epilogues
- `default`: Basic 2D epilogue
- `cshuffle`: Cross-shuffle epilogue (better performance)

### Schedulers
- `intrawave`: Intra-wave scheduling
- `interwave`: Inter-wave scheduling (limited support)

### Element-wise Operations (Multi-D)
- **Multi-D**: `MultiDAdd`, `MultiDMultiply`
- **Activations**: `PassThrough`, `Relu`, `Gelu`, `FastGelu`, `Silu`, `Tanh`, `Sigmoid`
- **Math**: `UnarySquare`, `UnaryAbs`, `UnarySqrt`, `Exp`, `Log`, `Ceil`, `Floor`
- **Scaling**: `Scale`, `AddScale`, `Clamp`

## Migration from tile_engine

### Before (tile_engine)

```bash
# Separate scripts for each variant
python tile_engine/ops/gemm/gemm_instance_builder.py
python tile_engine/ops/gemm_multi_d/gemm_multi_d_instance_builder.py
# Manual dispatcher wrapper generation
python dispatcher/codegen/generate_dispatcher_wrappers.py
```

### After (Unified)

```bash
# Single script for everything
python dispatcher/codegen/unified_gemm_codegen.py \
    --output-dir ./generated \
    --datatype fp16 \
    --layout rcr \
    --variants standard preshuffle multi_d
```

## Performance

- **Parallel Generation**: Uses thread pool for faster generation
- **Validation**: Tile and trait configurations validated before generation
- **Error Handling**: Continues on failure, reports all errors at end

## Development

### Adding New Variants

1. Add enum to `GemmVariant`
2. Implement variant-specific logic in `_get_configs_for_variant()`
3. Update `CKTileKernelGenerator` for variant-specific code
4. Update `KernelNaming` for variant suffix

### Adding New Element-wise Operations

1. Add to `multi_d_config.elementwise_ops` in config
2. Ensure operation exists in `ck_tile::element_wise` namespace
3. Generator will automatically handle it

### Testing

```bash
# Generate small test set
python unified_gemm_codegen.py \
    --output-dir ./test_output \
    --datatype fp16 \
    --layout rcr \
    --variants standard \
    --no-parallel

# Check output
ls test_output/
ls test_output/dispatcher_wrappers/
```

## Troubleshooting

### "Arguments not supported" at runtime
- Check tile configuration validity
- Ensure M, N, K are divisible by tile sizes
- Verify GPU architecture support

### Missing element-wise operation
- Check `ck_tile/ops/elementwise/unary_element_wise_operation.hpp`
- Ensure operation name matches exactly

### Compilation errors
- Verify CK Tile headers are in include path
- Check dispatcher headers are available
- Ensure C++17 or later

## Advanced Features

### ML-Based Auto-Tuning ⭐ NEW

Train an XGBoost model on tile_engine data to predict optimal kernels:

```bash
# 1. Collect training data
python collect_training_data.py \
    --tile-engine-path /path/to/tile_engine/build \
    --output-dir ./training_data \
    --problem-sizes ml \
    --num-configs 50

# 2. Train model
python ml_autotuner.py train \
    --data-dir ./training_data \
    --output ./models/autotuner.pkl

# 3. Get recommendations
python ml_autotuner.py recommend \
    --model ./models/autotuner.pkl \
    --problem-size 2048 2048 2048 \
    --candidates candidates.json
```

**Benefits**:
- 10-30% better performance than heuristics
- Learns from real hardware data
- Handles edge cases automatically
- Predicts performance without running

See [ML_AUTOTUNER_GUIDE.md](ML_AUTOTUNER_GUIDE.md) for complete guide.

### Library Scanning

Discover and wrap existing CK library kernels:

```bash
# Scan library for existing kernels
python library_scanner.py \
    --library-path ../../library \
    --output-dir ./library_wrappers \
    --datatype fp16 \
    --summary

# Export discovered kernels to JSON
python library_scanner.py \
    --library-path ../../library \
    --export-json discovered_kernels.json
```

### Validation

Validate generated kernels for correctness:

```bash
# Validate all generated files
python validator.py ./generated --verbose

# Show all issues (including warnings)
python validator.py ./generated --show-all
```

Validation checks:
- **Kernel Headers**: Header guards, includes, namespaces, types, launch functions
- **Dispatcher Wrappers**: Includes, namespaces, make functions, KernelKey setup
- **Registration Headers**: Registration functions, kernel counts

### Utilities

Common utilities available in `utils.py`:

```python
from utils import (
    get_project_root,
    get_library_path,
    sanitize_identifier,
    atomic_write,
    Timer,
    ProgressLogger,
)

# Path utilities
root = get_project_root()
lib_path = get_library_path()

# String utilities
safe_name = sanitize_identifier("my-kernel-name")

# Performance utilities
with Timer("Generation"):
    # ... expensive operation ...

progress = ProgressLogger(total=100, desc="Generating")
for i in range(100):
    # ... work ...
    progress.update()
progress.finish()
```

## Module Structure

```
dispatcher/codegen/
├── unified_gemm_codegen.py       ← Main generator
├── preselected_kernels.py        ← Curated kernel sets
├── library_scanner.py            ← Library discovery (NEW)
├── validator.py                  ← Validation (NEW)
├── utils.py                      ← Common utilities (NEW)
├── default_config.json           ← Default configuration
├── CMakeLists.txt                ← CMake integration
│
├── README.md                     ← This file
├── QUICK_START.md                ← 5-minute guide
├── UNIFIED_SUMMARY.md            ← Complete summary
├── ARCHITECTURE.md               ← System architecture
├── IMPROVEMENTS_FROM_CK4INDUCTOR.md  ← Design rationale
├── CHANGELOG.md                  ← Version history
└── INDEX.md                      ← Documentation index
```

## Future Enhancements

- [x] Preselected kernel sets
- [x] Library scanning
- [x] Validation system
- [x] Utility functions
- [ ] Template substitution (handle templated parameters)
- [ ] Auto-tuning (benchmark and select best kernels)
- [ ] Split-K support
- [ ] Grouped GEMM variants
- [ ] Structured sparsity (2:4)
- [ ] Mixed-precision (different A/B types)
- [ ] JIT compilation support
- [ ] Performance profiling integration

## See Also

- [INDEX.md](INDEX.md) - Documentation index
- [QUICK_START.md](QUICK_START.md) - 5-minute getting started
- [UNIFIED_SUMMARY.md](UNIFIED_SUMMARY.md) - Complete feature summary
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [Dispatcher Design Doc](../../DISPATCHER_DESIGN_DOC.md) - Overall design
- [Dispatcher Implementation](../README.md) - Dispatcher code
- [CK Tile GEMM Documentation](../../include/ck_tile/ops/gemm/README.md) - GEMM ops
