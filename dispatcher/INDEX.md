# CK Tile Dispatcher - File Index

Quick reference to all files in the dispatcher module.

---

## 📖 Documentation (Start Here)

| File | Purpose |
|------|---------|
| [README.md](README.md) | Main overview and quick start |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute getting started guide |
| [BUILD_AND_TEST.md](BUILD_AND_TEST.md) | Complete build and test instructions |
| [VALIDATION.md](VALIDATION.md) | Test results and validation report |
| [../DISPATCHER.md](../DISPATCHER.md) | Complete design specification |

---

## 🔧 Core Implementation

### Headers (`include/ck_tile/dispatcher/`)
| File | Purpose |
|------|---------|
| `dispatcher.hpp` | Main dispatcher class |
| `registry.hpp` | Kernel registry (thread-safe) |
| `kernel_key.hpp` | Kernel configuration metadata |
| `problem.hpp` | Problem specification |
| `kernel_instance.hpp` | Abstract kernel interface |

### Backend Wrappers (`include/ck_tile/dispatcher/backends/`)
| File | Purpose |
|------|---------|
| `generated_tile_backend.hpp` | For unified_gemm_codegen.py kernels ⭐ |
| `tile_backend.hpp` | For tile_engine style kernels |
| `kernel_registration.hpp` | Registration helpers |
| `backend_base.hpp` | Backend abstractions |

### Implementation (`src/`)
| File | Purpose |
|------|---------|
| `dispatcher.cpp` | Dispatcher implementation |
| `registry.cpp` | Registry implementation |

---

## 🐍 Python Integration

### Python API (`python/`)
| File | Purpose |
|------|---------|
| `dispatcher_api.py` | High-level Python API ⭐ |
| `bindings.cpp` | pybind11 C++ bindings |
| `__init__.py` | Package interface |
| `core.py` | Core types |
| `config.py`, `utils.py` | Utilities |

---

## 🧪 Tests

### C++ Tests (`test/`) - 51 tests, 100% passing
| File | Tests |
|------|-------|
| `test_kernel_key.cpp` | 7 tests - KernelKey functionality |
| `test_problem.cpp` | 5 tests - Problem validation |
| `test_registry.cpp` | 8 tests - Registry operations |
| `test_dispatcher.cpp` | 14 tests - Dispatcher selection |
| `test_tile_backend.cpp` | 6 tests - Backend integration |
| `test_integration_e2e.cpp` | 11 tests - End-to-end workflows |
| `test_mock_kernel.hpp` | Testing utilities |

### Python Tests (`python/tests/`)
| File | Purpose |
|------|---------|
| `test_cpp_bindings.py` | C++ extension validation |
| `test_core.py` | High-level API tests |

---

## 📝 Examples

| File | Purpose |
|------|---------|
| `single_tile_kernel_example.cpp` | Real CK Tile kernel GPU execution ⭐ |
| `python_complete_workflow.py` | Python API demonstration ⭐ |
| `python_gpu_example.py` | C++ extension usage |

---

## 🛠️ Code Generation

### Scripts (`codegen/`)
| File | Purpose |
|------|---------|
| `unified_gemm_codegen.py` | Main kernel generator ⭐ |
| `generate_dispatcher_registration.py` | Auto-registration code gen |
| `preselected_kernels.py` | Curated kernel sets |
| `validator.py` | Kernel validation |
| `utils.py` | Common utilities |

### Configs (`codegen/`)
| File | Purpose |
|------|---------|
| `default_config.json` | Default kernel configurations |
| `minimal_test_config.json` | Test configuration |

### Scripts (`codegen/`)
| File | Purpose |
|------|---------|
| `generate_test_kernels.sh` | Convenience script |

---

## 🏗️ Build System

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Main build configuration |
| `test/CMakeLists.txt` | Test build configuration |
| `python/CMakeLists.txt` | Python extension build |
| `examples/CMakeLists.txt` | Example builds |
| `codegen/CMakeLists.txt` | Codegen integration |

---

## 🔄 Generated Files (build/)

### Kernels (`build/generated_kernels/`)
- `gemm_*.hpp` - Generated CK Tile kernel headers
- `registration/dispatcher_registration.hpp` - Auto-registration code
- `registration/kernels_manifest.json` - Kernel metadata

### Build Artifacts (`build/`)
- `libck_tile_dispatcher.a` - C++ library
- `_dispatcher_native.so` - Python extension
- `examples/single_tile_kernel_example` - GPU executable

---

## 📊 File Count Summary

- **Documentation:** 4 essential guides
- **C++ Headers:** 12 files
- **C++ Implementation:** 2 files
- **C++ Tests:** 7 files (51 individual tests)
- **Python API:** 8 files
- **Codegen:** 7 scripts + 2 configs
- **Examples:** 3 working examples
- **Build System:** 5 CMakeLists.txt

**Total: ~50 essential files** (cleaned from 60+)

---

## 🎯 Quick Navigation

**Want to...**
- **Get started quickly?** → [QUICKSTART.md](QUICKSTART.md)
- **Build and test?** → [BUILD_AND_TEST.md](BUILD_AND_TEST.md)
- **See test results?** → [VALIDATION.md](VALIDATION.md)
- **Understand design?** → [../DISPATCHER.md](../DISPATCHER.md)
- **Use Python API?** → `python/dispatcher_api.py`
- **See working example?** → `examples/single_tile_kernel_example.cpp`
- **Generate kernels?** → `codegen/unified_gemm_codegen.py`

---

**Maintained by:** CK Tile Team  
**License:** MIT  
**Last Updated:** February 4, 2025

