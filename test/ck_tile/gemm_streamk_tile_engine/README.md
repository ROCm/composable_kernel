# Stream-K GEMM Tile Engine Unit Tests

## How It Works

This unit test system integrates **tile_engine's kernel generation** into automated testing:

1. **Uses tile_engine scripts directly**: Same Python scripts that generate tile_engine kernels
2. **JSON-based configuration**: Define test parameters in JSON files (like tile_engine)
3. **Build-time generation**: CMake calls tile_engine scripts to generate kernel headers
4. **Individual test executables**: Each kernel configuration becomes a separate test
5. **Tile_engine verification**: Uses exact same error thresholds and validation as tile_engine

## Tile Engine Integration

```
JSON Config → tile_engine Python scripts → Generated Headers → Test Executables
```

- **`--list_kernels`**: Get available kernel configurations from JSON
- **`--gen_individual`**: Generate all kernel headers in parallel during CMake configuration
- **`--gen_single`**: Generate individual kernel header for each configuration  
- **Same verification**: Uses tile_engine's adaptive error thresholds and reference calculations
- **Same patterns**: Follows tile_engine's tensor initialization, stride calculation, and kernel launching

### Config-Specific Test Parameters

Each test configuration can specify optimized problem sizes in its JSON file:
- **`test_params.problem_sizes`**: Array of `{m, n, k, split_k}` configurations
- **CMake extraction**: `extract_test_params.py` generates config-specific test parameter files
- **Build integration**: Each test target uses parameters appropriate for its kernel configuration
- **Optimized testing**: Different configs test different problem sizes that showcase their strengths


The key idea: **Unit tests that use tile_engine's exact kernel generation and verification methodology** instead of creating separate test infrastructure.

## Test Configurations

### 1. **Simple Test** (`simple_test_config.json`)
- **Purpose**: Basic functionality validation for fp16/bf16 data types
- **Config**: 128x128x32, warp 2x2x1, warp_tile 32x32x16  
- **Traits**: compv3 pipeline only
- **Coverage**: All 4 layouts (rcr, rrr, ccr, crr) for  fp16, bf16

## Data Type Support
- ✅ **fp16, bf16**: Fully supported - all layouts (rcr, rrr, ccr, crr)
- ❌ **fp64**: Not supported (hardware MFMA limitation)
- ⏳ **fp32, bf8, pk-int4-t**: Not yet supported by gemm_instance_builder (will be added later)

## Test Result Behavior

Tests automatically handle unsupported configurations through runtime validation:
- **PASSED**: Kernel executed correctly with results within error thresholds ✅
- **SKIPPED**: Kernel validation returned "Arguments not supported" (expected for certain problem sizes/configurations) ⚠️
- **FAILED**: Actual error or incorrect computation results ❌

When a kernel's `IsSupportedArgument()` check fails (e.g., due to vector alignment requirements, dimension constraints, or padding limitations), the test is automatically skipped rather than failed. This allows comprehensive testing across various problem sizes while gracefully handling configurations that don't meet specific kernel requirements.
