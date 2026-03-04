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
Test configs are generated during the Generation Phase. They are stored under the build directory at test/ck_tile/gemm_streamk_tile_engine/configs. The Compute Unit (CU) count of the device is required to generate the configs. If the Generation Phase occurs on a machine without a GPU or does not contain same GPU architecture on which you will run the tests, you can manually set the CU count using the `CU_COUNT` option:
```bash
# Assuming you are at the root of the repo
cd build
../script/cmake-ck-dev.sh .. gfx90a  -G Ninja -DCU_COUNT=100
```
You can reference the public whitepaper for your specific GPU to get the appropriate CU count. 
If no `CU_COUNT` option is given and no HIP device is found, then the default value of 100 CUs will be used to determine the problem sizes tested.

### 1. **Smoke Tests**
- **Purpose**: Basic functionality validation for fp16/bf16/fp8/bf8 data types
- **Config**: 256x256x32 (for bf16/fp16) or 128x128x32 (for bf8/fp8), warp 2x2x1, warp_tile 32x32x16  
- **Traits**: compv3 pipeline only
- **Coverage**: All 4 layouts (rcr, rrr, ccr, crr)

## Data Type Support
- ✅ **fp16, bf16, fp8, bf8**: Fully supported - all layouts (rcr, rrr, ccr, crr)
- ❌ **fp64**: Not supported (hardware MFMA limitation)
- ⏳ **fp32, pk-int4-t**: Not yet supported by gemm_instance_builder (will be added later)

## Test Result Behavior

Tests automatically handle unsupported configurations through runtime validation:
- **PASSED**: Kernel executed correctly with results within error thresholds ✅
- **SKIPPED**: Kernel validation returned "Arguments not supported" (expected for certain problem sizes/configurations) ⚠️
- **FAILED**: Actual error or incorrect computation results ❌

When a kernel's `IsSupportedArgument()` check fails (e.g., due to vector alignment requirements, dimension constraints, or padding limitations), the test is automatically skipped rather than failed. This allows comprehensive testing across various problem sizes while gracefully handling configurations that don't meet specific kernel requirements.
