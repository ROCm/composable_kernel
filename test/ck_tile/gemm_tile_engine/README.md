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
- **`--gen_single`**: Generate individual kernel header for each configuration  
- **Same verification**: Uses tile_engine's adaptive error thresholds and reference calculations
- **Same patterns**: Follows tile_engine's tensor initialization, stride calculation, and kernel launching




The key idea: **Unit tests that use tile_engine's exact kernel generation and verification methodology** instead of creating separate test infrastructure.
