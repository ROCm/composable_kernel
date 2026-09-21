# CShuffleLds LDS Microbenchmarks

Microbenchmark suite for measuring LDS (Local Data Share) bandwidth and bank conflicts in the CShuffleEpilogue cross-lane shuffle patterns.

## What This Measures

The CShuffleEpilogue uses LDS to redistribute GEMM output tiles from MFMA register layout to thread-raked layout for efficient global memory writes. This benchmark isolates the LDS store/load operations to measure:

1. **Store bandwidth** - Writing accumulator tiles to LDS (MFMA → LDS)
2. **Load bandwidth** - Reading shuffled tiles from LDS (LDS → thread-raked)
3. **Bank conflicts** - LDS bank conflicts during store/load (via rocprofv3)

## Configurations

Benchmarks are generated for all combinations of:

- **FP32 MFMA tiles**: 32x32x4, 32x32x8, 16x16x4, 16x16x8, 16x16x16
- **FP16 MFMA tiles**: 32x32x8, 32x32x16, 16x16x16, 4x64x16, 64x4x16
- **FP8 MFMA tiles**: 32x32x16, 16x16x32 (output FP16 or FP8)
- **Wave layouts**: 4x1, 2x2, 1x4 (block size = MFMA tile × wave layout)

**gfx950-only configurations:**
- **FP16**: 16x16x32
- **BF16**: 16x16x64 (uses gfx950-only 16x16x32 base instruction)
- **FP8**: 32x32x32, 32x32x64, 16x16x64, 16x16x128 (output FP16 or FP8)

Each configuration produces two measurements: Store and Load.

## Building

```bash
cmake -G Ninja -B build -S . \
  -DGPU_TARGETS=gfx950 \
  -DBUILD_CK_EXAMPLES=ON \
  -DBUILD_CK_TILE_CSHUFFLE_LDS_BENCHMARKS=ON

ninja -C build bench_lds_fp8_16x16x128_2x2_fp8  # Single benchmark
```

## Running

```bash
# Run a single benchmark
./build/bin/bench_lds_fp8_16x16x128_2x2_fp8 --warmup 3 --iters 10

# Profile with rocprofv3 for bank conflicts
cat > counters.txt <<EOF
pmc: SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS
EOF

rocprofv3 -i counters.txt -d output/ -- \
  ./build/bin/bench_lds_fp8_16x16x128_2x2_fp8
```

## Implementation

- **Generic kernels**: `include/ck_tile/utility/tile_load_store_microkernels.hpp`
- **Setup adapters**: `benchmark_cshuffle_lds.hpp`
- **Template generation**: `benchmark_template.cpp.in`

The benchmark uses CK's `launch_kernel` infrastructure for timing and `make_kernel` for functor-based kernel dispatch.
