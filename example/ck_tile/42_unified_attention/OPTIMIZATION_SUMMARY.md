# CK Unified Attention Optimization Summary

## Overview

Optimized the CK (Composable Kernel) unified attention kernel for d64 GQA-8 (DeepSeek-V3/R1 config: 64 query heads, 8 KV heads, head_dim=64) on MI350 (gfx950).

**Result: CK wins ~68% of shapes and is 1-4% faster than Triton end-to-end on production traces.**

| Metric | Before | After |
|--------|--------|-------|
| CK-winning shapes | 100/363 (27.5%) | **~248/363 (68%)** |
| Decode (weighted) | CK 36% slower | **CK 4-6% faster** |
| Prefill (weighted) | CK 36% slower | **CK ~tied or slightly faster** |
| Worst-case ratio | 3.55x slower | **~1.2x** |

---

## Optimization 1: Single Warp Group Serial Pipeline

**Problem:** The original pipeline required `NumWarpGroups == 2` (8 warps, 512 threads), wasting resources for decode with small Q tiles.

**Fix:** Relaxed the assertion and added a serial pipeline path for `NumWarpGroups == 1`:

```cpp
// unified_attention_pipeline.hpp
constexpr index_t NumWarpGroups = Problem::kBlockSize / Policy::NumThreadPerWarpGroup;
static_assert(NumWarpGroups == 1 || NumWarpGroups == 2);

// ...
if constexpr(NumWarpGroups == 1)
{
    // Serial pipeline: load V → PV GEMM → load K → QK GEMM → softmax
    // No warp group interleaving needed
}
```

Key constraint discovered: `kv_tile` is a **union** (K and V share registers), so PV GEMM must finish before K is loaded.

**Impact:** Enabled 4-warp and 2-warp decode kernels. ~1.7x speedup on 64-seq decode.

---

## Optimization 2: Async Prefetch Overlap

**Problem:** The serial pipeline loaded K/V synchronously, then computed, with no overlap.

**Fix:** Issue next iteration's global→LDS copies immediately after the barrier, overlapping with current GEMM compute:

```cpp
// Start next K/V loads right after barrier (overlap with compute below)
if(i_total_loops + 1 < num_total_loop)
    K_mem_load(number<1>{}); // async: next K → LDS
V_mem_load(number<0>{}); // async: next V → LDS

// Current iteration compute (overlaps with async loads above)
V_lds_load(number<1>{}); // read current V from LDS
fmha_alu1(number<0>{}); // softmax
gemm(number<0>{}, number<1>{}); // PV GEMM
K_lds_load(number<0>{}); // read current K from LDS
gemm(number<0>{}, number<0>{}); // QK GEMM
```

**Impact:** ~5% speedup on decode.

---

## Optimization 3: 2-Warp Decode Kernel (kBlockM=64)

**Problem:** 4-warp kernel with kBlockM=128 and kBlockQ=16 wastes 15/16 Q tile rows for decode.

**Fix:** Created `UnifiedAttentionPipelineDecodePolicy` with `NumWarpPerGroup=2`, enabling `sequence<2,1,1>` (2 warps):

```cpp
struct UnifiedAttentionPipelineDecodePolicy : UnifiedAttentionPipelineDefaultPolicy
{
    static constexpr ck_tile::index_t NumWarpPerGroup = 2;
    static constexpr ck_tile::index_t NumThreadPerWarpGroup =
        NumWarpPerGroup * ck_tile::get_warp_size();
};
```

kBlockM=64, kBlockQ=8 for GQA-8. Reduced tile waste from 15/16 to 7/8.

**Impact:** Additional ~5% on decode.

---

## Optimization 4: Early Exit + 2D Decode Grid

**Problem:** The 1D grid with binary search (`find_seq_idx`) had overhead and padding blocks.

**Fix:** For pure decode, use `dim3(num_kv_heads, num_seqs)` detected by `gridDim.y > 1`:

```cpp
// unified_attention_kernel.hpp
CK_TILE_HOST static constexpr auto GridSizeDecode(index_t num_kv_heads, index_t num_seqs)
{
    return dim3(num_kv_heads, num_seqs);
}

CK_TILE_DEVICE void operator()(Kargs kargs) const
{
    if(gridDim.y > 1)
    {
        // Direct mapping: no binary search, no padding CTAs
        kv_head_idx = blockIdx.x;
        seq_idx     = blockIdx.y;
    }
    else
    {
        // Standard 1D grid with binary search
        // ...
    }
}
```

Also moved the early-exit check before LDS allocation and binary search.

**Impact:** ~3% on high-batch decode.

---

## Optimization 5: 16x16 MFMA Tiny Decode (kBlockM=16, kBlockQ=2)

**Problem:** With 32x32 MFMA, minimum kBlockM=32 (1 warp), kBlockQ=4. Triton uses BLOCK_Q=2.

**Fix:** Use 16x16x32 MFMA instruction with `sequence<16,16,32>` warp tile. The softmax `permlane32_swap` reduction assumes 32x32 MFMA lane layout, so added a conditional fallback:

```cpp
// unified_attention_pipeline.hpp
static constexpr ck_tile::index_t kWarpGemmM =
    UnifiedAttentionShape::Gemm0WarpTile::at(ck_tile::number<0>{});

// In fmha_alu0 and fmha_alu1:
#if defined(__gfx950__)
if constexpr(kWarpGemmM == 32)
{
    // permlane32_swap for 32x32 MFMA (2 lanes per row)
    int32x2_t swapped_regs = __builtin_amdgcn_permlane32_swap(...);
    m_latest.thread_buf_[0] = f_max(swapped_regs.x, swapped_regs.y);
}
else
{
    // Generic reduction for 16x16 MFMA (4 lanes per row)
    block_tile_reduce_sync(m_latest, f_max, bool_constant<false>{});
}
#endif
```

New traits with `TinyDecodePolicy` (`NumWarpPerGroup=1`):

```cpp
struct unified_attention_decode_tiny_kernel_traits
{
    static constexpr index_t kBlockM = 16;
    static constexpr index_t BLOCK_SIZE = 64; // kPageBlockSize
    using unified_attention_warp_gemm_shape = sequence<16, 16, 32>;
    using unified_attention_block_warps     = sequence<1, 1, 1>;
    // ...
};
```

**Impact:** This was the breakthrough. CK went from 37% to 68% win rate. Matches Triton's BLOCK_Q=2 exactly.

---

## Optimization 6: 4-Tier Dispatch Heuristic

**Problem:** Single kernel config for all shapes.

**Fix:** Shape-adaptive dispatch based on average query length:

```cpp
static tile_tier select_tile_tier(const unified_attention_args& args)
{
    const index_t avg_q = args.num_seqs > 0 ? args.num_tokens / args.num_seqs : args.num_tokens;

    if(avg_q <= 2)   return tile_tier::tiny;   // 1 warp, 16x16 MFMA, kBlockM=16
    if(avg_q <= 8)   return tile_tier::small;  // 2 warps, kBlockM=64
    return tile_tier::medium;                   // 4 warps, kBlockM=128 (all prefill)
}
```

Verified by exhaustive sweep: 4-warp kBlockM=128 outperforms 8-warp kBlockM=256 on **all 71 prefill shapes** (0 exceptions).

**Impact:** 15-45% improvement on prefill shapes.

---

## Kernel Configurations

| Tier | Warps | MFMA | kBlockM | kBlockQ (GQA-8) | Policy | Use Case |
|------|-------|------|---------|-----------------|--------|----------|
| Tiny | 1 | 16x16x32 | 16 | 2 | TinyDecode | Pure decode (avg_q ≤ 2) |
| Small | 2 | 32x32x16 | 64 | 8 | Decode | Short decode (avg_q ≤ 8) |
| Medium | 4 | 32x32x16 | 128 | 16 | Default | All prefill |
| Large | 8 | 32x32x16 | 256 | 32 | Default | Unused (4-warp always better) |

---

## Instance Files

20 instance files covering d64/d128 × bf16/fp16 × mask/nomask × decode tiers:

```
instances/unified_attention_d64_bf16_mask_gqa8.cpp          # prefill (medium)
instances/unified_attention_d64_bf16_mask_gqa8_decode.cpp   # small decode
instances/unified_attention_d64_bf16_mask_gqa8_decode_s.cpp # small decode (2D grid)
instances/unified_attention_d64_bf16_mask_gqa8_decode_t.cpp # tiny decode (16x16 MFMA)
# ... (same pattern for bf16_nmask, fp16_mask, fp16_nmask, d128 variants)
```

---

## What Didn't Work

| Attempt | Why it failed |
|---------|--------------|
| kBlockM=64 with 2x2 warp layout | `permlane32_swap` assumes 1D warp layout; 2D breaks softmax reduction |
| 1-warp kBlockM=32 (32x32 MFMA) | Reduced memory bandwidth (1 warp) cancelled the tile waste savings |
| sp buffer 2→1 | VGPRs stayed at 132 (compiler minimum); slight decode regression from changed scheduling |
| kBlockPerCu=4 | `__launch_bounds__` hint didn't force VGPR reduction on ROCm |
| LDS padding changes | Inter-warp padding irrelevant for 1-warp; intra-warp conflicts from MFMA access pattern |
| kPageBlockSize=32 | 88 VGPRs / 5 waves, but 2x more KV iterations → 27% slower on low-batch decode |
| FMHA develop branch | Standard FMHA fwd kernel 4.6x slower than our decode kernel on 64-seq |

---

## Profile (512-seq decode, MI350/gfx950)

| Resource | Value | Limit | Occupancy |
|----------|-------|-------|-----------|
| VGPRs | 132 | 512/SIMD | 3 waves/SIMD |
| LDS | 38 KB | 160 KB/CU | 4 WGs/CU |
| Threads/WG | 64 (1 warp) | - | - |
| LDS bank conflicts | 17.8M | - | Intra-warp pattern |

Bottleneck: VGPRs (132 is compiler minimum for kPageBlockSize=64 with 16x16 MFMA).

---

## Files Modified

**Pipeline:**
- `include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp` — serial pipeline, async prefetch, 16x16 MFMA reduction
- `include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline_default_policy.hpp` — decode policies

**Kernel:**
- `include/ck_tile/ops/unified_attention/kernel/unified_attention_kernel.hpp` — 2D decode grid, early exit

**Dispatch:**
- `example/ck_tile/42_unified_attention/unified_attention.cpp` — 4-tier dispatch
- `example/ck_tile/42_unified_attention/unified_attention_impl.hpp` — decode kernel traits

**Instances:**
- `example/ck_tile/42_unified_attention/instances/` — 12 new decode instance files

**aiter JIT:**
- `aiter/jit/optCompilerConfig.json` — registered decode instance files
