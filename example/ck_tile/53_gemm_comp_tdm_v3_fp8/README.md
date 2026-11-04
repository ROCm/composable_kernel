# comp_tdm_v3 fp8 GEMM benchmark (gfx1250)

This example times the `GemmPipelineAgBgCrCompTDMV3` pipeline on gfx1250. It also checks the results. The benchmark is meant for a same-session comparison against the rocKE fp8 GEMM `8febcf8e5c` at the lead shape M = N = 4096, K = 65536.

**Status:**

- The pipeline uses its default policy: one split barrier per K tile, with the B sub-tile held in registers and A read one row slice at a time. The example is built with LLVM's post-RA scheduler on (`-mllvm -enable-post-misched=1`, see Build), like the comp_tdm_v3 gtests. The target is at least 0.98x of `8febcf8e5c` (peak protocol, random data, bf16 output, same host and session).
- The numerics were checked under the gfx1250 emulator (rocJITsu `mirage`), with TheRock 10.1.0 and 10.2.0. An earlier form of this pipeline has run correctly on gfx1250 A0; this form has not yet run on hardware.
- Earlier 16-wave TDM pipelines have hung on hardware. Run the 2048³ check (step 1 below) before anything else.

## The kernel

| Item | Value |
|---|---|
| Data types | A, B fp8 e4m3 (OCP); fp32 accumulate; C bf16 |
| Layout | RCR: A row-major M x K, B column-major (N x K), C row-major M x N; packed strides |
| Block tile | 256 x 256 x 256, 16 waves (warps 4 x 4 x 1), 512 threads, 1 block per CU |
| Warp tile | 32 x 32 x 128 (WMMA f8f6f4) |
| Pipeline | `GemmPipelineAgBgCrCompTDMV3`, default policy (alternating TDM issuers), double LDS buffer |
| Epilogue | CShuffle, bf16 out |
| Partitioner | `GemmTile1DPartitioner` |
| Limits | M, N and K must be multiples of 256 (no padding); split-K 1 |
| Resources (both compilers) | 249 VGPRs, 0 scratch, 278,464 B LDS, occupancy 4 waves/SIMD |

## Requirements

- A ROCm/TheRock install whose clang accepts `--offload-arch=gfx1250`. Tested with TheRock 10.1.0 (AMD clang 24.0.0git, llvm-project `d6f6cb6`, the CK CI compiler) and the TheRock 10.2.0 prerelease (llvm-project `4f43f47`). Do not use TheRock 10.0 (clang 23) for gfx1250.
- Source: the PR branch `users/afagaj/ck-tile-gfx1250-fp8-gemm` of ROCm/rocm-libraries (rebased as it is updated), or a frozen snapshot of it, `users/afagaj/ck-tile-gfx1250-fp8-gemm-SNAPSHOT-<n>`.

## Build

Option 1, CMake target (run from the rocm-libraries checkout):

```bash
export ROCM_PATH=/path/to/therock/rocm
cmake -S projects/composablekernel -B build-ck \
  -DCMAKE_PREFIX_PATH=$ROCM_PATH -DROCM_PATH=$ROCM_PATH \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1250 -DBUILD_TESTING=OFF -DBUILD_CK_EXAMPLES=ON
make -C build-ck tile_example_gemm_comp_tdm_v3_fp8_bench
# binary: build-ck/bin/tile_example_gemm_comp_tdm_v3_fp8_bench
```

Option 2, one clang++ command. It uses the device flags of CK's CMake build. `-Rpass-analysis` prints the VGPR, scratch and LDS numbers; please send them back.

```bash
CK=projects/composablekernel; $ROCM_PATH/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1250 -std=c++20 -O3 -DNDEBUG -DCK_CMAKE_GPU_TARGET_IDS=0x1250 -DCK_ENABLE_BF16 -DCK_ENABLE_BF8 -DCK_ENABLE_FP16 -DCK_ENABLE_FP32 -DCK_ENABLE_FP64 -DCK_ENABLE_FP8 -DCK_ENABLE_INT8 -DCK_GFX1250_SUPPORT -DCK_GFX12_SUPPORT -DCK_TILE_USE_OCP_FP8 -DCK_TILE_USE_WMMA=1 -DCK_TIME_KERNEL=1 -DCK_USE_GFX1250 -DCK_USE_NATIVE_MX_SUPPORT -DCK_USE_OCP_FP8 -DCK_USE_WMMA -DCK_USE_WMMA_FP8 -DCK_USE_XDL -DUSE_NEW_UNIFIED_FRAMEWORK=0 -DARCH_GFX12 -fbracket-depth=1024 -fno-offload-uniform-block -mllvm --lsr-drop-solution=1 -mllvm -enable-post-misched=1 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -mllvm -enable-noalias-to-md-conversion=0 -Rpass-analysis=kernel-resource-usage -I$CK/include $CK/example/ck_tile/53_gemm_comp_tdm_v3_fp8/gemm_comp_tdm_v3_fp8_bench.cpp -o ck_fp8_bench > ck_fp8_bench.build.log 2>&1
```

## Arguments

Arguments use CK's `-name=value` form. `-?` prints the list.

| Argument | Default | Meaning |
|---|---|---|
| `-m`, `-n`, `-k` | 4096, 4096, 65536 | problem size (multiples of 256) |
| `-mode` | `peak` | `verify-int`, `verify-rand`, `peak`, `sustained` |
| `-data` | `rand` | input data for `peak` / `sustained`: `rand` or `int` |
| `-seed` | 12345 | input RNG seed (the same seed gives the same bytes in every mode) |
| `-sigma` | 0.1 | `rand` data: normal(0, sigma), then rounded to e4m3 (RNE, saturating) |
| `-warmup`, `-repeat` | peak 1, 3; sustained 50, 100 | untimed and timed launches per sample |
| `-rounds` | peak 15; sustained 1 | samples |
| `-gap_ms` | peak 25; sustained 0 | idle time before each sample |
| `-verify_launches` | 5 | verify modes: launches; every output must be byte-identical to the first |
| `-check` | `auto` | verify modes: `full`, `sample`, or `auto` (full when M·N·K ≤ 2^33) |
| `-check_per_tile` | 64 | sampled check: random points per 256 x 256 output tile, plus the tile's 4 corners |

### Modes

- **`verify-int`** fills A and B with integers in [-5, 5], which e4m3 represents exactly. Every partial sum is an integer below 2^24, so fp32 accumulation is exact in any order. C must equal the exact sum rounded once to bf16 (RNE), **bit for bit**.
- **`verify-rand`** uses the same random data as `peak`. It compares against a double-precision host reference, with the per-element tolerance `|got - ref| <= |ref|·2^-7 + Σ|a·b|·2^-16`. It reports:
  - `max_abs_err`;
  - `max_rel_err`, which is `|got - ref| / |ref|` over points where `|ref|` ≥ 1% of the RMS of the reference. The bf16 rounding alone allows up to 2^-8 ≈ 3.9e-3.
- **Reference cost.** At 4096 x 4096 x 65536 a full CPU reference costs 1.1e12 MACs, so `auto` checks a sample there: 64 random points per output tile plus each tile's corners, 17,408 points in about a second. The sample catches tile-wide and systematic errors, not a single wrong element. `-check=full` forces the full check (CPU-bound and threaded; it can take tens of minutes).
- **`peak`** runs `-rounds` samples. Each sample is: idle `-gap_ms`, `-warmup` untimed launches, then `-repeat` launches between two `hipEvent`s. The sample value is the mean time per launch. The defaults (warmup 1, repeat 3, 25 ms idle, 15 samples, median) follow the rocKE peak protocol. At 4096 x 4096 x 65536 a 0.4 ms launch gives a timed window of about 1.2 ms. If this kernel is much slower, `-repeat=1` keeps the window near 1.2 ms.
- **`sustained`** runs 50 warmup and 100 timed launches back to back, with no idle time.
- **Cache and input data.** No cache flush or rotating buffers are used. TFLOPS = 2·M·N·K / time.

### Output

Every line that starts with `{` is JSON:

- `"type":"info"`: kernel name, device, arch, CU count, HIP version, compiler.
- `"type":"verify"`: `checked`, `bad`, `nondeterministic_launches`, errors, `result` PASS/FAIL.
- `"type":"sample"`: one per timed sample (`ms`, `tflops`).
- `"type":"summary"`: `median_ms`, `best_ms`, `mean_ms`, `median_tflops`, `best_tflops`, `best_over_median`.

The exit code is 0 on success, 1 on a verify FAIL, and 2 to 5 on argument, HIP or host-encoder errors.

## Run sequence (same session and host as the `8febcf8e5c` run)

Wait until no other GPU process has run for about a minute, and keep it that way during the runs.

```bash
B=./ck_fp8_bench   # or build-ck/bin/tile_example_gemm_comp_tdm_v3_fp8_bench
# 1. Smoke and hang check first (16-wave hang history). If it does not finish within a minute, kill it and report.
timeout 120 $B -m=2048 -n=2048 -k=2048 -mode=verify-int                  # full check, bit-exact
# 2. Verify at the lead shape (sampled check, 5 launches, outputs must be identical)
$B -m=4096 -n=4096 -k=65536 -mode=verify-int
$B -m=4096 -n=4096 -k=65536 -mode=verify-rand
# 3. Timing, random data. Interleave with 8febcf8e5c the way you time your own arms.
$B -m=4096 -n=4096 -k=65536 -mode=peak        # 15 samples: warmup 1, repeat 3, 25 ms idle each
$B -m=4096 -n=4096 -k=65536 -mode=sustained   # warmup 50, repeat 100, back to back
```

To interleave per sample with your harness, call `$B -mode=peak -rounds=1` once per round and take the median yourself. One process per sample adds host fill time (about 512 MiB of fp8 encoding) before the idle gap, so the idle is longer than 25 ms. That is still the peak regime.

The random data is normal(0, 0.1) rounded to e4m3, which matches our reading of your "randn·0.1". If your GEMM inputs use another distribution, tell us, or set `-sigma`. WMMA throughput depends on the data, so the CK and rocKE runs must use the same distribution. `-data=int` times on the integer inputs.

## What to send back

1. The full stdout of every command above (the JSON lines are enough), and the build log lines with the kernel's VGPRs, ScratchSize and LDS Size.
2. Your `8febcf8e5c` numbers from the same session: peak median (and protocol) and sustained, on the same random data.
3. The ratio CK / `8febcf8e5c` at peak and at sustained, and best/median for each arm.
4. Environment:
   - host name;
   - `rocminfo` ASIC revision (stepping);
   - `$ROCM_PATH/.info/version`;
   - `clang --version`;
   - rocm-libraries commit;
   - whether the GPU process list stayed clean;
   - your roofline probe before and after, if you ran it.
