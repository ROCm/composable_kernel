# Unified Attention — Compile-Time Parameter Reference

All values are derived from the kernel traits structs in `unified_attention_impl.hpp`,
the shape/problem/pipeline/policy headers under `include/ck_tile/ops/unified_attention/`,
and the dispatch logic in `unified_attention.cpp`.

## Kernel Trait Variants

There are five kernel-traits structs, each targeting a different workload profile:

| Traits Struct | Use Case | Default HeadSize | Default BlockM | Default NumQPerKV | Default BlockSize |
|---|---|---|---|---|---|
| `unified_attention_kernel_traits` | Prefill (large Q) | 128 | 256 | 1 | 32 (64 if HeadSize≤64) |
| `unified_attention_decode_kernel_traits` | Decode medium | 128 | 128 | 1 | 32 (64 if HeadSize≤64) |
| `unified_attention_decode_small_kernel_traits` | Decode small | 64 | 64 | 8 | 64 |
| `unified_attention_decode_tiny_kernel_traits` | Decode tiny | 64 | 16 | 8 | 64 |
| `unified_attention_decode_bs32_kernel_traits` | Decode bs32 narrow | 64 | 32 | 8 | 32 |

---

## Resolved Parameter Values Per Variant

### 1. Prefill — `unified_attention_kernel_traits` (default: d128, MHA)

| Parameter | Value | Source |
|---|---|---|
| **HeadSize (kHeadDim)** | 128 | Template arg `HeadSize_` |
| **kHeadDimPadded** | 128 | `ceil_to_qualified_tile_length<128>()` = 128 (power of two) |
| **kBlockM** | 256 | Template arg `BlockM_` |
| **NumQueriesPerKV** | 1 | Template arg `NumQPerKV_` |
| **kBlockQ** | 256 | `kBlockM / num_queries_per_kv` = 256/1 |
| **kPageBlockSize (BLOCK_SIZE)** | 32 | `BlockTile::at<2>` (HeadSize > 64 → 32) |
| **kBlockSize (threads)** | 512 | `NumWarps * WarpSize` = 8 × 64 |
| **NumWarps** | 8 | `max(NumGemm0Warps, NumGemm1Warps)` = max(8,8) |
| **BlockWarps (Gemm0 & Gemm1)** | `<8, 1, 1>` | `unified_attention_block_warps` |
| **WarpGemmShape (Gemm0 & Gemm1)** | `<32, 32, 16>` | `unified_attention_warp_gemm_shape` |
| **IsVLayoutRowMajor** | true | Shape template arg |
| **kPadSeqLenQ** | true | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDim** | false | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDimQ** | false | Pipeline: `kPadHeadDimQ = Problem::kPadHeadDim` |
| **kPadHeadDimV** | false | Pipeline: `kPadHeadDimV = Problem::kPadHeadDim` |
| **kBlockPerCu** | 2 | Traits sets -1 → pipeline defaults to 2 |
| **NumWarpGroups** | 2 | `kBlockSize / NumThreadPerWarpGroup` = 512/256 |
| **Policy::NumWarpPerGroup** | 4 | `UnifiedAttentionPipelineDefaultPolicy` |
| **Policy::NumThreadPerWarpGroup** | 256 | 4 × 64 |
| **Policy::kKLdsPadInBytes** | 16 | 4 × 4 (4 dwords) |
| **Policy::kVLdsPadInBytes** | 64 | 4 × 16 (16 dwords) |
| **Data types** | fp16 or bf16 (Q/K/V/P/O), float (Sacc/Oacc/LSE) | Problem traits |

**Gemm0 (Q×K):** M=256, N=32, K=128, warps=`<8,1,1>`, warp_tile=`<32,32,16>`
**Gemm1 (P×V):** M=256, N=128, K=32, warps=`<8,1,1>`, warp_tile=`<32,32,16>`

---

### 2. Decode Medium — `unified_attention_decode_kernel_traits` (default: d128, MHA)

| Parameter | Value | Source |
|---|---|---|
| **HeadSize (kHeadDim)** | 128 | Template arg |
| **kHeadDimPadded** | 128 | Power of two |
| **kBlockM** | 128 | Template arg |
| **NumQueriesPerKV** | 1 | Template arg |
| **kBlockQ** | 128 | 128/1 |
| **kPageBlockSize (BLOCK_SIZE)** | 32 | HeadSize > 64 → 32 |
| **kBlockSize (threads)** | 256 | 4 × 64 |
| **NumWarps** | 4 | `max(4, 4)` |
| **BlockWarps** | `<4, 1, 1>` | 4 warps along M |
| **WarpGemmShape** | `<32, 32, 16>` | |
| **kPadSeqLenQ** | true | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDim** | false | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDimQ** | false | Pipeline: `kPadHeadDimQ = Problem::kPadHeadDim` |
| **kPadHeadDimV** | false | Pipeline: `kPadHeadDimV = Problem::kPadHeadDim` |
| **NumWarpGroups** | 1 | 256/256 |
| **Policy** | `UnifiedAttentionPipelineDefaultPolicy` (NumWarpPerGroup=4) | |
| **kBlockPerCu** | 2 | Default |

**Gemm0 (Q×K):** M=128, N=32, K=128
**Gemm1 (P×V):** M=128, N=128, K=32

---

### 3. Decode Small — `unified_attention_decode_small_kernel_traits` (default: d64, GQA-8)

| Parameter | Value | Source |
|---|---|---|
| **HeadSize (kHeadDim)** | 64 | Template arg |
| **kHeadDimPadded** | 64 | Power of two |
| **kBlockM** | 64 | Template arg |
| **NumQueriesPerKV** | 8 | Template arg (GQA-8) |
| **kBlockQ** | 8 | 64/8 |
| **kPageBlockSize (BLOCK_SIZE)** | 64 | HeadSize ≤ 64 → 64 |
| **kBlockSize (threads)** | 128 | 2 × 64 |
| **NumWarps** | 2 | `max(2, 2)` |
| **BlockWarps** | `<2, 1, 1>` | 2 warps along M |
| **WarpGemmShape** | `<32, 32, 16>` | |
| **kPadSeqLenQ** | true | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDim** | false | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDimQ** | false | Pipeline: `kPadHeadDimQ = Problem::kPadHeadDim` |
| **kPadHeadDimV** | false | Pipeline: `kPadHeadDimV = Problem::kPadHeadDim` |
| **NumWarpGroups** | 1 | 128/128 (NumWarpPerGroup=2) |
| **Policy** | `UnifiedAttentionPipelineDecodePolicy` (NumWarpPerGroup=**2**) | |
| **kBlockPerCu** | 2 | Default |

**Gemm0 (Q×K):** M=64, N=64, K=64
**Gemm1 (P×V):** M=64, N=64, K=64

---

### 4. Decode Tiny — `unified_attention_decode_tiny_kernel_traits` (default: d64, GQA-8)

| Parameter | Value | Source |
|---|---|---|
| **HeadSize (kHeadDim)** | 64 | Template arg |
| **kHeadDimPadded** | 64 | Power of two |
| **kBlockM** | 16 | Template arg |
| **NumQueriesPerKV** | 8 | Template arg (GQA-8) |
| **kBlockQ** | 2 | 16/8 |
| **kPageBlockSize (BLOCK_SIZE)** | 64 | HeadSize ≤ 64 → 64 |
| **kBlockSize (threads)** | 64 | 1 × 64 |
| **NumWarps** | 1 | `max(1, 1)` |
| **BlockWarps** | `<1, 1, 1>` | 1 warp |
| **WarpGemmShape** | `<16, 16, 32>` | **16×16 MFMA** (different from other tiers) |
| **kPadSeqLenQ** | true | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDim** | false | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDimQ** | false | Pipeline: `kPadHeadDimQ = Problem::kPadHeadDim` |
| **kPadHeadDimV** | false | Pipeline: `kPadHeadDimV = Problem::kPadHeadDim` |
| **NumWarpGroups** | 1 | 64/64 (NumWarpPerGroup=1) |
| **Policy** | `UnifiedAttentionPipelineTinyDecodePolicy` (NumWarpPerGroup=**1**) | |
| **kBlockPerCu** | 2 | Default |

**Gemm0 (Q×K):** M=16, N=64, K=64
**Gemm1 (P×V):** M=16, N=64, K=64

---

### 5. Decode BS32 Narrow — `unified_attention_decode_bs32_kernel_traits` (default: d64, GQA-8, BS=32)

| Parameter | Value | Source |
|---|---|---|
| **HeadSize (kHeadDim)** | 64 | Template arg |
| **kHeadDimPadded** | 64 | Power of two |
| **kBlockM** | 32 | Template arg |
| **NumQueriesPerKV** | 8 | Template arg (GQA-8) |
| **kBlockQ** | 4 | 32/8 |
| **kPageBlockSize (BLOCK_SIZE)** | 32 | Explicit template arg |
| **kBlockSize (threads)** | 128 | 2 × 64 |
| **NumWarps** | 2 | `max(2, 2)` |
| **BlockWarps** | `<2, 1, 1>` | 2 warps along M |
| **WarpGemmShape** | `<16, 16, 32>` | 16×16 MFMA |
| **kPadSeqLenQ** | true | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDim** | false | `TileUnifiedAttentionTraits<true, false, -1>` |
| **kPadHeadDimQ** | false | Pipeline: `kPadHeadDimQ = Problem::kPadHeadDim` |
| **kPadHeadDimV** | false | Pipeline: `kPadHeadDimV = Problem::kPadHeadDim` |
| **NumWarpGroups** | 1 | 128/128 (NumWarpPerGroup=2) |
| **Policy** | `UnifiedAttentionPipelineDecodePolicy` (NumWarpPerGroup=**2**) | |
| **kBlockPerCu** | 2 | Default |

**Gemm0 (Q×K):** M=32, N=32, K=64
**Gemm1 (P×V):** M=32, N=64, K=32

---

## Dispatched Instances (from `unified_attention.cpp`)

### d128, MHA (`num_queries_per_kv == 1`)

Always uses the **prefill** tier (8 warps, kBlockM=256):

| Data Type | Masking | Traits | HeadSize | BlockM | NQPKV | kBlockQ | Threads |
|---|---|---|---|---|---|---|---|
| fp16 | no | `kernel_traits` | 128 | 256 | 1 | 256 | 512 |
| fp16 | yes | `kernel_traits` | 128 | 256 | 1 | 256 | 512 |
| bf16 | no | `kernel_traits` | 128 | 256 | 1 | 256 | 512 |
| bf16 | yes | `kernel_traits` | 128 | 256 | 1 | 256 | 512 |

### d64, GQA-8 (`num_queries_per_kv == 8`), `page_blk_size >= 64`

Tier selected by `select_tile_tier()` based on average and max query length:

| Tier | Condition | Traits | HeadSize | BlockM | kBlockQ | Warps | Threads | MFMA | Grid |
|---|---|---|---|---|---|---|---|---|---|
| **Tiny** | avg_q ≤ 2, max_q ≤ 2 | `decode_tiny` | 64 | 16 | 2 | 1 | 64 | 16×16 | decode 2D |
| **Small** | avg_q ≤ 8, max_q ≤ 8 | `decode_small` | 64 | 64 | 8 | 2 | 128 | 32×32 | decode 2D |
| **Medium** | avg_q ≤ 16, max_q ≤ 16 | `decode` | 64 | 128 | 16 | 4 | 256 | 32×32 | standard 1D |
| **Large** | otherwise | `kernel_traits` | 64 | 256 | 32 | 8 | 512 | 32×32 | standard 1D |

### d64, GQA-8 (`num_queries_per_kv == 8`), `page_blk_size < 64` (BS32 variants)

| Tier | Traits | HeadSize | BlockM | kBlockQ | Warps | Threads | MFMA | Grid |
|---|---|---|---|---|---|---|---|---|
| **Tiny** | `decode_bs32` | 64 | 32 | 4 | 2 | 128 | 16×16 | decode 2D |
| **Small** | `decode_small` (BS=32) | 64 | 64 | 8 | 2 | 128 | 32×32 | decode 2D |
| **Medium** | `decode` (BS=32) | 64 | 128 | 16 | 4 | 256 | 32×32 | standard 1D |

---

## Tier Selection Logic

```
avg_q = num_tokens / num_seqs
max_q = max_seqlen_q (or avg_q if 0)

kBlockQ_tiny  = 16 / num_queries_per_kv   (= 2 for GQA-8)
kBlockQ_small = 64 / num_queries_per_kv   (= 8 for GQA-8)
kBlockQ_medium = 128 / num_queries_per_kv (= 16 for GQA-8)

if avg_q ≤ kBlockQ_tiny  AND max_q ≤ kBlockQ_tiny  → tiny
if avg_q ≤ kBlockQ_small AND max_q ≤ kBlockQ_small → small
otherwise                                           → medium
```

The **large** tier (prefill, 8 warps) is only dispatched for `kernel_traits` directly —
it is not reachable through `select_tile_tier()` (which only returns tiny/small/medium).
The large tier is effectively the d128-MHA path or used when no decode tier matches.

---

## Policy Parameters Summary

| Policy | NumWarpPerGroup | NumThreadPerWarpGroup | Used By |
|---|---|---|---|
| `DefaultPolicy` | 4 | 256 | Prefill, Decode Medium |
| `DecodePolicy` | 2 | 128 | Decode Small, Decode BS32 Narrow |
| `TinyDecodePolicy` | 1 | 64 | Decode Tiny |

All policies share:
- `kKLdsPadInBytes = 16` (4 dwords between warps in K LDS)
- `kVLdsPadInBytes = 64` (16 dwords between warps in V LDS)
- `SmemKPackK = 16 / sizeof(DataType)` → 8 for fp16/bf16
- `SmemVPackK = 16 / sizeof(DataType)` → 8 for fp16/bf16
- Block GEMM type: `BlockGemmARegBRegCRegV2` (A/B in registers, C in registers)
- LDS K/V buffer count: 4 (quad-buffered, `GetSmemSize = 4 * GetSmemSizeKV`)

---

## Shape Struct Breakdown (`TileUnifiedAttentionShape`)

The `BlockTile` sequence encodes four values:

```
sequence<kBlockM, kBlockQ, kPageBlockSize, kHeadDim>
```

| Field | Meaning |
|---|---|
| `kBlockM` | Tile along the flattened batch dimension (num_queries_per_kv × q_seqlen_tile) |
| `kBlockQ` | Tile along q seqlen only (= kBlockM / num_queries_per_kv) |
| `kPageBlockSize` | Tile along K/V seqlen dimension (BLOCK_SIZE for paged KV cache) |
| `kHeadDim` | Head dimension |
| `kHeadDimPadded` | `ceil_to_qualified_tile_length(kHeadDim)` — rounds to supported tile size |

---

## Grid Dimensions

| Grid Mode | Formula | Used By |
|---|---|---|
| **Standard 1D** | `dim3(num_kv_heads * total_num_q_blocks)` | Prefill, Medium decode |
| **Decode 2D** | `dim3(num_kv_heads, num_seqs)` | Small/Tiny decode |

Where `total_num_q_blocks = num_tokens / kBlockQ + num_seqs`.

---

## Padding Flags Explained

Three related flags control out-of-bounds handling:

| Flag | Defined In | Meaning |
|---|---|---|
| `kPadSeqLenQ` | Traits → Problem | If true, Q/O tile windows are padded along the seqlen_q dimension so loads/stores beyond the actual sequence length read zeros. All example variants set this to **true**. |
| `kPadHeadDim` | Traits → Problem | Master switch for head-dimension padding. If true, Q/K/V/O tiles are padded from `kHeadDim` up to `kHeadDimPadded` with zeros. All example variants set this to **false** (head dims used are exact powers of two so no padding needed). |
| `kPadHeadDimQ` | Pipeline | Alias: `Problem::kPadHeadDim`. Controls whether Q and K tile views are padded along the head dimension. When false, vector load alignment (`kAlignmentQ/K`) can use the full natural vector width; when true alignment is forced to 1. |
| `kPadHeadDimV` | Pipeline | Alias: `Problem::kPadHeadDim`. Same as above but for V tile views and `kAlignmentV`. |

The alignment impact:

```
kAlignmentQ = kPadHeadDimQ ? 1 : Policy::GetAlignmentQ<Problem>()
kAlignmentK = kPadHeadDimQ ? 1 : Policy::GetAlignmentK<Problem>()
kAlignmentV = kPadHeadDimV ? 1 : Policy::GetAlignmentV<Problem>()
kAlignmentO = kPadHeadDimV ? 1 : Policy::GetAlignmentO<Problem>()
```

When padding is off (the case for all dispatched instances), the pipeline can use wider
vector loads (e.g. 128-bit / 8 elements for fp16), which is critical for memory throughput.

---

## LDS (Shared Memory) Size — `GetSmemSize()` Explained

The pipeline's `GetSmemSize()` determines total LDS allocation per workgroup:

```cpp
static constexpr index_t GetSmemSize()
{
    return max(kBlockM * kHeadDimPadded * sizeof(PDataType),
               Policy::GetSmemSize<Problem>() +
                   kBlockM * kPageBlockSize * sizeof(PDataType));
}
```

This computes the **maximum** of two LDS usage scenarios that share the same memory
at different phases of the pipeline:

### Scenario A: Output accumulator in LDS
```
kBlockM × kHeadDimPadded × sizeof(PDataType)
```
Used when the output accumulator tile (`o_acc`, shape `kBlockM × kHeadDimPadded`) is
temporarily stored to LDS — e.g. for cross-warp-group reduction or for the epilogue
to read back and write to global memory.

### Scenario B: KV buffers + P (softmax output) in LDS simultaneously
```
Policy::GetSmemSize<Problem>()  +  kBlockM × kPageBlockSize × sizeof(PDataType)
```
- **`Policy::GetSmemSize`** = `4 × GetSmemSizeKV` — quad-buffered K and V LDS tiles
  used for async-copy pipelining (2 buffers for K, 2 for V, each double-buffered).
- **`kBlockM × kPageBlockSize × sizeof(PDataType)`** — the P tile (softmax output,
  shape `kBlockM × kPageBlockSize`) that must live in LDS at the same time as the
  KV buffers, because Gemm1 (P×V) reads P from LDS while V is also in LDS.

The `max()` takes whichever phase needs more, since they reuse the same LDS allocation.

### Concrete values (fp16/bf16, `sizeof(PDataType) = 2`):

| Variant | kBlockM | kHeadDimPadded | kPageBlockSize | Scenario A | Policy KV (4 bufs) | P tile | Scenario B | **Total LDS** |
|---|---|---|---|---|---|---|---|---|
| Prefill (d128) | 256 | 128 | 32 | 64 KiB | ~64 KiB* | 16 KiB | ~80 KiB | ~80 KiB |
| Decode Med (d128) | 128 | 128 | 32 | 32 KiB | ~32 KiB* | 8 KiB | ~40 KiB | ~40 KiB |
| Decode Small (d64) | 64 | 64 | 64 | 8 KiB | ~16 KiB* | 8 KiB | ~24 KiB | ~24 KiB |
| Decode Tiny (d64) | 16 | 64 | 64 | 2 KiB | ~8 KiB* | 2 KiB | ~10 KiB | ~10 KiB |
| Decode BS32 (d64) | 32 | 64 | 32 | 4 KiB | ~8 KiB* | 2 KiB | ~10 KiB | ~10 KiB |

\* Policy KV sizes are approximate; exact values include per-warp LDS padding
(`kKLdsPadInBytes`=16, `kVLdsPadInBytes`=64) which add a few KiB depending on
the number of warps and issue count.

---

## GEMM Dimension Mapping (`MPerBlock` / `NPerBlock` / `kKPerBlock`)

The policy functions use local variables named `kNPerBlock`, `kKPerBlock`, etc.
These are **not independent parameters** — they are GEMM-convention aliases (M=rows,
N=cols, K=reduction) for the existing shape constants, and their meaning **changes
depending on which operation** is being described.

### Gemm0: S = Q × K^T

| GEMM dim | Shape param | Meaning | Prefill | Decode Small |
|---|---|---|---|---|
| M | `kBlockM` | Flattened query tile (tokens × GQA heads) | 256 | 64 |
| N | `kPageBlockSize` | KV seqlen tile | 32 | 64 |
| K | `kHeadDim` | Head dimension (reduction) | 128 | 64 |

### Gemm1: O = P × V

| GEMM dim | Shape param | Meaning | Prefill | Decode Small |
|---|---|---|---|---|
| M | `kBlockM` | Same flattened query tile | 256 | 64 |
| N | `kHeadDim` | Head dimension (output) | 128 | 64 |
| K | `kPageBlockSize` | KV seqlen tile (reduction) | 32 | 64 |

Note that `kPageBlockSize` and `kHeadDim` **swap roles** between Gemm0 and Gemm1
(N↔K), because the seqlen dimension is the output of Q×K^T but the reduction
dimension of P×V.

### In policy code: K/V data-movement functions

These load K and V tiles shaped `[kPageBlockSize, kHeadDim]` from global memory
into LDS. Here the naming follows the **physical tile layout**, not any particular
GEMM's convention:

```
kNPerBlock = kPageBlockSize   (rows: positions along KV seqlen)
kKPerBlock = kHeadDim         (cols: head dimension, contiguous in memory)
```

### In policy code: V register distribution (Gemm1 perspective)

When building the V register tile for Gemm1 (P×V), the naming flips to Gemm1's
convention where V is the B-matrix:

```
kNPerBlock = kHeadDim         (Gemm1 output dim)
kKPerBlock = kPageBlockSize   (Gemm1 reduction dim)
```

### In pipeline code: `MakeSimpleLdsDesc<MPerBlock, NPerBlock>()`

`MPerBlock` and `NPerBlock` are just template parameters. The function is called as:

| Call site | `MPerBlock` = | `NPerBlock` = | Tile |
|---|---|---|---|
| S/P LDS window | `kBlockM` | `kPageBlockSize` | Attention scores / softmax output |
| O LDS window | `kBlockM` | `kHeadDimPadded` | Output accumulator |
| m/l LDS window (1D) | `kBlockM` | — | Row-wise max / sum for softmax |

---

## HeadDim Padding (`ceil_to_qualified_tile_length`)

| Input HeadDim | Padded HeadDim |
|---|---|
| 48 | 48 |
| 64 | 64 |
| 96 | 128 |
| 128 | 128 |
| 160 | 256 |
| 192 | 192 |
| 256 | 256 |
| Other power-of-two | Same |
