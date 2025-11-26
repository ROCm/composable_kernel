# CK Tile Dispatcher Examples

Practical examples demonstrating CK Tile Dispatcher usage.

> **See also:** [Main Dispatcher README](../README.md) for installation, build, and core concepts.

## Quick Start

```bash
cd /workspace/workspace/composable_kernel/dispatcher

# Build examples
mkdir -p build && cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DBUILD_DISPATCHER_EXAMPLES=ON \
  -DGPU_TARGETS=gfx942
make -j$(nproc)

# Run C++ example
./examples/example_01_basic_gemm

# Run Python example
cd ../examples/python
python3 01_basic_gemm.py
```

## C++ Examples (`cpp/`)

| Example | Description | Complexity |
|---------|-------------|------------|
| `01_basic_gemm.cpp` | Complete explicit workflow: KernelConfig → Registry → Dispatcher | ★☆☆☆☆ |
| `02_multi_size.cpp` | Multiple problem sizes | ★★☆☆☆ |
| `03_benchmark.cpp` | Performance testing with warmup | ★★★☆☆ |
| `04_validation.cpp` | Correctness vs CPU reference | ★★★☆☆ |
| `05_heuristics.cpp` | Kernel selection strategies | ★★★★☆ |
| `06_json_export.cpp` | Export registry to JSON | ★★☆☆☆ |
| `07_preshuffle.cpp` | PreShuffle pipeline | ★★★★☆ |
| `08_multi_d.cpp` | Multi-D GEMM with fusion | ★★★★★ |
| `09_multi_registry.cpp` | Multiple registries with different kernels | ★★★★★ |

### Running C++ Examples

```bash
cd build/examples

./example_01_basic_gemm              # Basic workflow
./example_03_benchmark 2048 2048 2048  # Benchmark specific size
./example_09_multi_registry          # Multiple registries
```

## Python Examples (`python/`)

| Example | Description | Complexity |
|---------|-------------|------------|
| `01_basic_gemm.py` | Complete workflow: KernelConfig → Registry → Dispatcher | ★☆☆☆☆ |
| `02_batch_gemm.py` | Multiple sizes via dispatcher | ★★☆☆☆ |
| `03_benchmark.py` | Performance testing | ★★★☆☆ |
| `04_validation.py` | Correctness vs NumPy | ★★★☆☆ |
| `05_numpy_integration.py` | GPUMatmul class | ★★☆☆☆ |
| `06_json_export.py` | Export registry to JSON | ★★☆☆☆ |
| `07_preshuffle.py` | PreShuffle kernel generation | ★★★★☆ |
| `08_multi_d.py` | Multi-D GEMM | ★★★★★ |
| `09_multi_registry.py` | Multiple registries with smart selection | ★★★★★ |

### Running Python Examples

```bash
cd examples/python

python3 01_basic_gemm.py     # Basic workflow
python3 04_validation.py     # Validate correctness
python3 09_multi_registry.py # Multiple registries
```

## Core Pattern

All examples follow the explicit data flow pattern:

```python
# Python
config = KernelConfig(tile_m=128, ...)  # 1. Define config
codegen.generate_from_config(config)     # 2. Generate kernel
registry = Registry(name="my_reg")       # 3. Create registry
registry.register_kernel(config)         # 4. Register config
dispatcher = Dispatcher(registry, lib)   # 5. Create dispatcher
result = dispatcher.run(A, B, M, N, K)   # 6. Run GEMM
```

```cpp
// C++
KernelKeyBuilder builder;                // 1. Build key
builder.tile_m = 128; ...
Registry::instance().register_kernel(k); // 2. Register kernel
Dispatcher dispatcher;                   // 3. Create dispatcher
dispatcher.run(a, b, c, problem);        // 4. Run GEMM
```

## Learning Path

1. **Start:** `01_basic_gemm` - Understand the complete workflow
2. **Scale:** `02_multi_size` / `02_batch_gemm` - Try different sizes
3. **Measure:** `03_benchmark` - Performance testing
4. **Verify:** `04_validation` - Correctness testing
5. **Integrate:** `05_numpy_integration` - Real-world usage
6. **Debug:** `06_json_export` - Export for analysis
7. **Optimize:** `07_preshuffle` - Advanced pipeline
8. **Fuse:** `08_multi_d` - Fused operations
9. **Scale:** `09_multi_registry` - Multiple registries for workloads

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Generated kernels not found" | Build with `-DBUILD_DISPATCHER_EXAMPLES=ON` |
| "HIP error" | Check GPU: `rocm-smi` |
| Low performance | Use larger sizes (4096+), Release build |
| Python import error | Set `PYTHONPATH` to include `dispatcher/python` |

---

> **More info:** See [../README.md](../README.md) for full documentation.
