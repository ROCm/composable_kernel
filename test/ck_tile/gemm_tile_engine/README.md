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
- **Coverage**: ~1 kernel per datatype/layout

### 3. **Large Datatype** (`large_datatype_config.json`)
- **Purpose**: Optimized for fp32 (fp64 not supported by hardware)
- **Config**: 64x64x16, warp 2x2x1, warp_tile 16x16x16
- **Traits**: compv3 pipeline only
- **Coverage**: ~1 kernel per datatype/layout

### 4. **Tile Size Coverage** (`tile_size_coverage_config.json`)
- **Purpose**: Test different tile dimensions (16-256 range)
- **Config**: Variable tile sizes, fixed warp config
- **Traits**: compv3 only
- **Coverage**: ~75 kernels per datatype/layout

### 5. **Traits Coverage** (`traits_coverage_config.json`)
- **Purpose**: Test all pipeline/epilogue/scheduler combinations
- **Config**: Fixed 64x64x32
- **Traits**: 3 pipelines × 2 epilogues × 2 schedulers × 2 persistent  
- **Coverage**: 24 kernels per datatype/layout

### 6. **Padding Coverage** (`padding_coverage_config.json`)
- **Purpose**: Test padding behavior (pad_m, pad_n, pad_k)
- **Config**: Fixed 64x64x32
- **Traits**: All padding combinations
- **Coverage**: 8 kernels per datatype/layout

## Data Type Support
- ✅ **fp16, bf16, fp8, fp32**: Fully supported
- ❌ **fp64**: Not supported (hardware MFMA limitation)
- ⏳ **pk-int4-t, bf8**: Not yet supported by gemm_instance_builder (will be added later)
