# CK Tile Dispatcher Examples

Comprehensive examples for GEMM and Convolution operations with GPU execution.

---

## Quick Start

### Step 1: Build

```bash
cd /path/to/composable_kernel/dispatcher
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  -DBUILD_DISPATCHER_EXAMPLES=ON

# Build everything (C++ examples + Python libraries)
make -j$(nproc)

# Or build ONLY Python libraries (faster)
make python_libs -j$(nproc)
```

### Step 2: Run C++ Examples

```bash
cd build/examples

# GEMM
./gemm_01_basic
./gemm_03_benchmark_validation

# Conv
./conv_01_basic
./conv_10_bwd_data --verify
./conv_11_bwd_weight --verify
```

### Step 3: Run Python Examples

```bash
cd /path/to/composable_kernel/dispatcher

# GEMM
python3 examples/gemm/python/01_basic_gemm.py
python3 examples/gemm/python/07_stress_test.py
python3 examples/gemm/python/08_heuristics.py

# Conv
python3 examples/conv/python/01_basic_conv.py
python3 examples/conv/python/04_conv2d_bwd_data.py --verify
```

---

## Directory Structure

```
examples/
├── gemm/
│   ├── cpp/           # 6 C++ GEMM examples
│   └── python/        # 11 Python GEMM examples
│
└── conv/
    ├── cpp/           # 11 C++ Conv examples
    └── python/        # 12 Python Conv examples
```

---

## GEMM Examples

### C++ Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | `gemm_01_basic` | Basic GEMM with declarative API, autofill, autocorrect |
| 02 | `gemm_02_multi_size` | Wildcard expansion for multiple configurations |
| 03 | `gemm_03_benchmark_validation` | Performance benchmarking with CPU/GPU validation |
| 04 | `gemm_04_heuristics` | Heuristic-based kernel selection |
| 05 | `gemm_05_json_export` | Registry JSON export for external tools |
| 06 | `gemm_06_multi_registry` | Multiple registries with named kernel sets |

**Details:** [gemm/cpp/README.md](gemm/cpp/README.md)

---

### Python Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | `01_basic_gemm.py` | Basic GEMM with multi-kernel support |
| 02 | `02_batch_gemm.py` | Batched GEMM operations |
| 03 | `03_benchmark.py` | Performance benchmarking |
| 04 | `04_validation.py` | CPU reference validation |
| 05 | `05_numpy_integration.py` | NumPy array integration |
| 06 | `06_json_export.py` | Registry JSON export |
| 07 | `07_stress_test.py` | Multi-kernel stress testing (48 configs) |
| 08 | `08_heuristics.py` | Heuristic-based kernel selection (24 configs) |
| 09 | `09_multi_registry.py` | Multiple registries |
| 10 | `10_advanced_benchmark.py` | Advanced benchmark with full control |
| 11 | `11_json_import.py` | Import kernels from JSON |

**Details:** [gemm/python/README.md](gemm/python/README.md)

---

## Convolution Examples

### C++ Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | `conv_01_basic` | Basic 2D forward convolution |
| 02 | `conv_02_forward` | Detailed 2D forward |
| 03 | `conv_03_validation` | CPU reference validation |
| 04 | `conv_04_multi_size` | Multiple problem sizes |
| 05 | `conv_05_benchmark` | Performance benchmarking |
| 06 | `conv_06_heuristics` | Heuristic kernel selection |
| 07 | `conv_07_json_export` | Registry JSON export |
| 08 | `conv_08_multi_registry` | Multiple registries |
| 09 | `conv_09_conv3d_forward` | 3D volumetric convolution |
| 10 | `conv_10_bwd_data` | Backward data gradient |
| 11 | `conv_11_bwd_weight` | Backward weight gradient |

**Details:** [conv/cpp/README.md](conv/cpp/README.md)

---

### Python Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | `01_basic_conv.py` | Basic 2D forward |
| 02 | `02_conv2d_fwd.py` | 2D forward patterns |
| 03 | `03_conv3d_fwd.py` | 3D forward patterns |
| 04 | `04_conv2d_bwd_data.py` | Backward data with validation |
| 05 | `05_conv2d_bwd_weight.py` | Backward weight with validation |
| 06 | `06_benchmark.py` | Performance benchmarking |
| 07 | `07_validation.py` | CPU vs GPU validation |
| 08 | `08_json_export.py` | Registry JSON export |
| 09 | `09_multi_registry.py` | Multiple registries |
| 10 | `10_conv3d_forward.py` | 3D conv with GPU |
| 11 | `11_bwd_data.py` | Backward data API |
| 12 | `12_bwd_weight.py` | Backward weight API |

**Details:** [conv/python/README.md](conv/python/README.md)

---

## Validation Examples

### C++ Validation

```bash
./gemm_03_benchmark_validation --verify 1    # GEMM with CPU reference
./gemm_03_benchmark_validation --verify 2    # GEMM with GPU reference
./conv_03_validation                         # Forward conv validation
./conv_10_bwd_data --verify                  # Backward data with CPU reference
./conv_11_bwd_weight --verify                # Backward weight with CPU reference
```

### Python Validation

```bash
python3 examples/gemm/python/04_validation.py
python3 examples/gemm/python/07_stress_test.py   # Multi-kernel validation
python3 examples/conv/python/04_conv2d_bwd_data.py --verify
python3 examples/conv/python/07_validation.py
```

---

## Key Features

### Declarative Kernel API

Both C++ and Python examples use a declarative approach:

**C++ (DECL_KERNEL_SET macro):**
```cpp
DECL_KERNEL_SET(my_kernels,
    .add(
        Signature().dtype("fp16").layout("rcr"),
        Algorithm().tile(256, 256, 32).wave(2, 2, 1).warp(32, 32, 16)
                   .pipeline("compv4").scheduler("intrawave"),
        "gfx942"
    )
);
```

**Python (KernelConfig):**
```python
config = KernelConfig(
    tile_m=256, tile_n=256, tile_k=32,
    wave_m=2, wave_n=2, wave_k=1,
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    pipeline="compv4", scheduler="intrawave"
)
```

### Autofill and Autocorrect

The build system automatically:
- **Autofills** missing parameters with sensible defaults
- **Autocorrects** invalid parameters based on architecture constraints
- **Expands** wildcards (`*`, `-1`, `ANY_INT`) to all valid configurations

### Architecture Filtering

Kernel configurations are validated against GPU architecture constraints:
- Tile divisibility requirements
- Warp tile constraints
- Pipeline compatibility

Invalid configurations are automatically pruned during code generation.

---

## Troubleshooting

### Python: Library not found

```bash
# Run from dispatcher directory
cd /path/to/composable_kernel/dispatcher
python3 examples/gemm/python/01_basic_gemm.py
```

### C++: Executables not found

```bash
# Build with examples enabled
cmake .. -DBUILD_DISPATCHER_EXAMPLES=ON
make -j$(nproc)

# Run from build/examples
cd build/examples
./gemm_01_basic
```

### GPU not detected

```bash
rocminfo | grep "Name:"
# Should show: gfx942, gfx90a, etc.
```
