# CK Tile Testing Guide

This document describes the test organization and available test targets for CK Tile operations.

## Overview

CK Tile tests are organized with multiple levels of granularity to support different development workflows:

1. **Global test labels** - Run tests across all operations
2. **Operation-specific umbrella targets** - Run all tests for a specific operation
3. **Individual test executables** - Run specific tests

## Global Test Labels

These targets run tests across all CK operations (not just CK Tile):

### `ninja smoke`
Run fast smoke tests (tests that complete within ~30 seconds on gfx90a).
```bash
ninja smoke
```

### `ninja regression`
Run slower, more comprehensive regression tests.
```bash
ninja regression
```

### `ninja check`
Run ALL available tests in the entire codebase.
```bash
ninja check
```

## Operation-Specific Umbrella Targets

These targets allow you to run all tests for a specific CK Tile operation. This is useful when making changes to a particular operation and wanting to validate all related tests without running the entire test suite.

### GEMM Operations

#### `ck_tile_gemm_tests`
Run all basic GEMM pipeline tests (memory, compute variants, persistent, etc.)
```bash
ninja ck_tile_gemm_tests
```
**Test executables included:**
- `test_ck_tile_gemm_pipeline_mem`
- `test_ck_tile_gemm_pipeline_compv3`
- `test_ck_tile_gemm_pipeline_compv4`
- `test_ck_tile_gemm_pipeline_persistent`
- `test_ck_tile_gemm_pipeline_compv6`
- `test_ck_tile_gemm_pipeline_comp_async` (gfx95 only)
- `test_ck_tile_gemm_pipeline_*_wmma` variants (gfx11/gfx12 only)

#### `ck_tile_gemm_block_scale_tests`
Run all GEMM tests with block-scale quantization (AQuant, BQuant, ABQuant, etc.)
```bash
ninja ck_tile_gemm_block_scale_tests
```
**Test executables included:** 29 test executables covering:
- AQuant tests (memory pipelines, base layouts, prefill, preshuffle, transpose)
- ABQuant tests (base, padding, preshuffle)
- BQuant tests (1D/2D variants, transpose)
- BQuant with PreshuffleB (decode/prefill, 1D/2D)
- BQuant with PreshuffleQuant (decode/prefill, 1D/2D)
- RowColQuant and TensorQuant tests

#### `ck_tile_gemm_streamk_tests`
Run all GEMM StreamK tests (tile partitioner, reduction, smoke, extended)
```bash
ninja ck_tile_gemm_streamk_tests
```
**Test executables included:**
- `test_ck_tile_streamk_tile_partitioner`
- `test_ck_tile_streamk_reduction`
- `test_ck_tile_streamk_smoke`
- `test_ck_tile_streamk_extended`

#### `ck_tile_grouped_gemm_quant_tests`
Run all grouped GEMM quantization tests
```bash
ninja ck_tile_grouped_gemm_quant_tests
```
**Test executables included:**
- `test_ck_tile_grouped_gemm_quant_rowcol`
- `test_ck_tile_grouped_gemm_quant_tensor`
- `test_ck_tile_grouped_gemm_quant_aquant`
- `test_ck_tile_grouped_gemm_quant_bquant`
- `test_ck_tile_grouped_gemm_quant_bquant_preshuffleb`

### Other Operations

#### `ck_tile_fmha_tests`
Run all FMHA (Flash Multi-Head Attention) tests
```bash
ninja ck_tile_fmha_tests
```
**Test executables included:** Forward and backward tests for fp16, bf16, fp8bf16, fp32

#### `ck_tile_reduce_tests`
Run all reduce operation tests
```bash
ninja ck_tile_reduce_tests
```
**Test executables included:**
- `test_ck_tile_reduce2d`
- `test_ck_tile_multi_reduce2d_threadwise`
- `test_ck_tile_multi_reduce2d_multiblock`

## Individual Test Executables

You can also build and run individual test executables:

### Build a specific test
```bash
ninja test_ck_tile_gemm_pipeline_mem
```

### Run a specific test directly
```bash
./build/bin/test_ck_tile_gemm_pipeline_mem
```

### Run a specific test through ctest
```bash
ctest -R test_ck_tile_gemm_pipeline_mem --output-on-failure
```






