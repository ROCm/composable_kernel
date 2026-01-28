# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Composable Kernel (CK) is AMD's high-performance GPU kernel library for machine learning workloads. It uses HIP C++ with a tile-based programming model and tensor coordinate transformation techniques.

**Two implementations exist:**
- **CK Tile** (`include/ck_tile/`) - Modern tile-programming API, preferred for new development
- **Legacy CK** (`include/ck/`) - Older implementation, still supported

## Build Commands

```bash
# Development build (from build directory)
mkdir build && cd build
../script/cmake-ck-dev.sh .. "gfx908;gfx90a;gfx942"
make -j32  # Use ~2GB RAM per thread

# Build specific targets
make tile_example_fmha_fwd    # FMHA forward example
make tile_example_fmha_bwd    # FMHA backward example
make ckProfiler               # Performance profiler

# Tests
make smoke      # Quick tests (<30s each)
make regression # Long tests (>=30s each)
make check      # All tests

# Single test
ctest -R "fmha" -V
```

**CMake options:**
- `GPU_TARGETS="gfx908;gfx90a;gfx942"` - Target GPU architectures
- `DTYPES="fp32;fp16;fp8;bf16;int8"` - Data types to build
- `BUILD_DEV=ON` - Development mode

## Code Formatting

CK uses clang-format. Install pre-commit hooks:
```bash
sudo script/install_precommit.sh
```

Disable temporarily with `git commit --no-verify`.

## Architecture

### Four-Layer Structure
1. **Templated Tile Operators** - Low-level tile operations
2. **Templated Kernel/Invoker** - Kernel templates with tile operators
3. **Instantiated Kernel/Invoker** - Concrete kernel instances
4. **Client API** - User-facing API

### Key Directories
- `include/ck_tile/core/` - Core utilities (containers, data types, coordinate transforms)
- `include/ck_tile/ops/` - Operator implementations (fmha, gemm, softmax, etc.)
- `include/ck_tile/ops/fmha/pipeline/` - FMHA pipeline implementations (performance-critical)
- `example/ck_tile/` - Working examples with build recipes
- `codegen/` - Python-based kernel code generation
- `profiler/` - Performance profiling tools

### FMHA (Flash Attention) Architecture

#### Directory Structure
```
include/ck_tile/ops/fmha/
├── kernel/           # Kernel entry points (fmha_fwd_kernel.hpp, fmha_fwd_v3_kernel.hpp)
├── pipeline/         # Pipeline implementations (performance-critical)
├── block/            # Block-level components (masking, dropout, position encoding)
└── api/              # High-level API wrappers
```

#### Kernel Template Structure

The kernel (`FmhaFwdKernel`, `FmhaFwdV3Kernel`) has two key template parameters:
- `FmhaPipeline` - Block tile pipeline handling Q*K and P*V computations
- `EpiloguePipeline` - Post-processing and output storage

**Key data types extracted from pipeline:**
- `QDataType`, `KDataType`, `VDataType` - Input types (fp8, fp16, bf16)
- `PDataType` - Attention probability type after softmax
- `SaccDataType` - Scratch accumulator (typically float)
- `ODataType` - Output type

**Configuration flags:**
- `kIsGroupMode` - Variable-length sequences via seqstart pointers
- `kPadSeqLenQ/K`, `kPadHeadDimQ/V` - Padding control
- `kHasLogitsSoftCap` - Gemma-style logits softcap
- `kStoreLSE` - Store log-sum-exp for backward pass
- `QScaleEnum` - FP8 quantization (PERTENSOR, NONE)

#### Pipeline Implementations

| Pipeline | Name | Description |
|----------|------|-------------|
| `BlockFmhaPipelineQRKSVS` | "qr" | LDS-based, all QKV in LDS. For medium sequences. |
| `BlockFmhaPipelineQRKSVSAsync` | "qr_async" | Q in registers, async K/V loading. For longer sequences. |
| `BlockFmhaFwdV3Pipeline` | "v3" | Next-gen with warp group coordination and instruction scheduling. |
| `BlockFmhaPipelineSplitKV` | - | Multi-pass with reduction for very long sequences. |
| `BlockFmhaPipelinePagedKV` | - | KV-cache paging for inference. |

#### Attention Computation Flow (Online Softmax)

```
Phase 1: GEMM0 (Q × K^T → S)
├── Load Q tile (M0 × K0) into registers
├── Loop over K tiles (N0 × K0):
│   ├── Async load K tile to LDS
│   ├── Sync barrier
│   └── Block GEMM with MFMA → S accumulator
└── Apply scale: S *= 1/sqrt(hdim)

Phase 2: Online Softmax
├── Row-wise max: m_j = max(S_j)
├── Optional: logits softcap (tanh transform)
├── Exponential: P = exp(S - m_j)
├── Row-wise sum: l_j = sum(P_j)
└── Rescale accumulator: O *= exp(m_old - m_new)

Phase 3: GEMM1 (P × V → O)
├── Convert P to compute type
├── Load V tiles (K1 × N1)
├── Block GEMM with MFMA → O accumulator
└── Finalize: O /= l_j

Phase 4: Epilogue
├── Convert O to output type
├── Optional: store LSE = m/log(2) + log(l)
└── Write O tile to DRAM
```

#### Memory Management

**LDS Layout:**
- K tiles: N0 × K0, double-buffered for async prefetch
- V tiles: K1 × N1, bank-conflict-aware padding
- Size computed via `Policy::GetSmemSize<Problem>()`

**Async Copy Pattern:**
```cpp
async_load_tile_raw(k_lds_window, k_dram_window);  // Non-blocking
move_tile_window(k_dram_window, {kN0, 0});
// ... GEMM computation overlaps with load ...
s_waitcnt_vmcnt<0>();  // Wait before use
```

**Prefetching Strategy:** Load K[i+1] while computing with K[i]

#### Block-Level Components

**Masking (`block_masking.hpp`):**
- `MASK_FROM_TOP_LEFT` - Causal (lower triangular)
- `MASK_FROM_BOTTOM_RIGHT` - Future tokens
- Local attention via `window_size_left/right`
- `GenericAttentionMask::GetTileRangeAlongX()` - Skip fully masked tiles

**Quantization (`block_attention_quant_scale_enum.hpp`):**
- `NONE` - Standard float operations
- `PERTENSOR` - Single scale per Q/K/V tensor (FP8)
- Flow: `Q_fp8 * scale → float → compute → saturate → O_fp8`

#### Policy/Trait Configuration

**TileFmhaTraits** - Core configuration:
```cpp
template <
    bool kPadSeqLenQ, kPadSeqLenK,
    bool kPadHeadDimQ, kPadHeadDimV,
    bool kHasLogitsSoftCap,
    BlockAttentionBiasEnum BiasEnum,
    bool kStoreLSE,
    bool kHasDropout,
    BlockAttentionQuantScaleEnum QScaleEnum
>
struct TileFmhaTraits;
```

**Default Policy** provides:
- Alignment hints for DRAM loads
- GEMM configurations (MFMA instruction selection)
- LDS store/load descriptors
- Register tile distributions

#### Grid/Block Organization

```cpp
dim3 GridSize(batch_size, nhead, ceil(max_seqlen_q / kM0) * ceil(hdim_v / kN1));
dim3 BlockSize(kBlockSize);  // Typically 256-512 threads
```

#### V3 Pipeline Optimizations

- **Warp Group Specialization** - 2 warp groups (4 waves each) with different roles
- **Phase Scheduling** - Explicit barriers for MFMA/VALU/TRANS timing
- **Packed FP32** - `v_pk_mul_f32` for two operations per instruction
- **Fast Exp2** - Bit manipulation approximation

## Key Concepts

- **Tile** - Fixed-size data chunk processed by a thread block
- **Block Tile** - Tile owned by entire thread block
- **Wave Tile** - Tile owned by a wavefront (64 threads on AMD)
- **LDS** - Local Data Share (AMD's shared memory)
- **MFMA** - Matrix Fused Multiply-Add (AMD's matrix core instruction)
- **XDL** - Crosslane Data Layout instructions

See `TERMINOLOGY.md` and `ACRONYMS.md` for complete references.

## Common Variable Naming

| Symbol | Meaning |
|--------|---------|
| M, N, K | GEMM dimensions: A[M,K] × B[K,N] = C[M,N] |
| Q, K, V | Query, Key, Value (attention) |
| S | Sequence length |
| D | Head dimension |
| B | Batch size |
| H | Number of attention heads |

## Running FMHA Examples

```bash
# Basic FMHA forward
./bin/tile_example_fmha_fwd -b=1 -h=16 -s=16384 -d=128

# With FP8
./bin/tile_example_fmha_fwd -b=1 -h=8 -s=4096 -d=128 -prec=fp8

# Group mode (variable length)
./bin/tile_example_fmha_fwd -mode=1 -b=2 -h=8 -s=1024,2048 -d=128

# With causal mask
./bin/tile_example_fmha_fwd -b=1 -h=8 -s=4096 -d=128 -mask=t
```

Use `-?` flag to see all options.

## Codegen System

Kernels are instantiated into separate files via Python scripts in `codegen/` to enable parallel compilation. Example: `example/ck_tile/01_fmha/codegen/generate.py`.
