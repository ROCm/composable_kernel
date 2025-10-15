# GEMM Tile Engine Unit Tests

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
- **Purpose**: Basic functionality validation  
- **Config**: 128x128x64, warp 2x2x1, warp_tile 16x16x16
- **Traits**: compv3 + compv4 pipelines
- **Coverage**: ~2 kernels per datatype/layout

### 2. **Small Datatype** (`small_datatype_config.json`)
- **Purpose**: Optimized for fp8/fp16/bf16 data types
- **Config**: 128x128x32, warp 2x2x1, warp_tile 32x32x16  
- **Traits**: compv3 pipeline only
- **Coverage**: 
  - fp16, bf16: All 4 layouts (rcr, rrr, ccr, crr)
  - fp8: RCR layout only (other layouts not approved)

### 3. **Large Datatype** (`large_datatype_config.json`)
- **Purpose**: Optimized for fp32
- **Config**: 64x64x16, warp 2x2x1, warp_tile 16x16x16
- **Traits**: compv3 pipeline only
- **Coverage**: RCR layout only (other layouts not approved)

### 4. **Tile Size Coverage** (Quick or Comprehensive)
- **Purpose**: Test different tile dimensions and warp configurations
- **Quick** (`tile_size_quick_config.json`): Less than 100 kernels
  - tile_m/n: [32, 64, 128, 256], tile_k: [16, 32, 64]
  - warp config: 2×2×1, warp_tile 16×16×16
  - Focused set for fast validation
- **Comprehensive** (`tile_size_comprehensive_config.json`): More than 1000 kernels
  - tile_m/n: [16-256 step 16]
  - tile_k: [16, 32, 64]
  - warp_m/n: [1, 2, 4], warp_tile_m/n: [16, 32], warp_tile_k: [16, 32]
  - Extensive coverage across multiple warp configurations and MFMA tile sizes
  - Exact count varies based on validation filtering
- **Traits**: compv3 pipeline only
- **Note**: Use CMake option `-DTILE_SIZE_LEVEL=comprehensive` to enable comprehensive testing (default is quick)

### 5. **Traits Coverage** (`traits_coverage_config.json`)
- **Purpose**: Test all pipeline/epilogue/scheduler combinations
- **Config**: Fixed 64x64x32
- **Traits**: 3 pipelines × 2 epilogues × 2 schedulers × 2 persistent  
- **Coverage**: 24 kernels per datatype/layout

### 6. **Padding Coverage** (`padding_coverage_config.json`)
- **Purpose**: Test padding behavior with all padding flags enabled
- **Config**: Fixed 64x64x32, warp 2x2x1, warp_tile 32x32x16
- **Padding**: All enabled (pad_m=true, pad_n=true, pad_k=true)
- **Problem sizes**: Vector-aligned but not tile-aligned (104×104×56, 200×152×80, 152×200×64)
- **Coverage**: 1 kernel configuration testing padding with irregular sizes

## Data Type Support
- ✅ **fp16, bf16**: Fully supported - all layouts (rcr, rrr, ccr, crr)
- 🟡 **fp8**: Supported - RCR layout only (other layouts not approved)
- 🟡 **fp32**: Supported - RCR layout only (other layouts not approved)
- ❌ **fp64**: Not supported (hardware MFMA limitation)
- ⏳ **pk-int4-t, bf8**: Not yet supported by gemm_instance_builder (will be added later)
