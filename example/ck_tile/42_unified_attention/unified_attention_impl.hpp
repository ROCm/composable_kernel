// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// fp8 d128 prefill: use the wide v_mfma_f32_32x32x64_f8f6f4 MMA (vs narrow
// 32x32x16). Default on; build with -DUA_FP8_WIDE_MMA=0 for the narrow A/B.
#ifndef UA_FP8_WIDE_MMA
#define UA_FP8_WIDE_MMA 1
#endif

#include <type_traits>
#include <utility>

#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/unified_attention/block/block_masking.hpp"
#include "ck_tile/ops/unified_attention/kernel/unified_attention_kernel.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline_problem.hpp"
#include "ck_tile/ops/unified_attention/pipeline/tile_unified_attention_shape.hpp"
#include "ck_tile/ops/unified_attention/pipeline/tile_unified_attention_traits.hpp"

#include "unified_attention.hpp"
#include "mask.hpp"

namespace ck_tile {

// =============================================================================
// KernelVariant
//
// Flat enum of every compiled kernel instance. Each variant fixes
// (kBlockM, warp count, MFMA shape, pipeline policy) via a variant_config<V>
// specialization below. This is the single source of truth for "what knobs
// differ between kernel instances".
//
// page_size is intentionally NOT part of this enum. The multi-page-tile fix
// in the pipeline decoupled kBlockN from page_blk_size, so every variant is
// correct for any page size.
// =============================================================================
enum class KernelVariant
{
    // d=128 (num_queries_per_kv chosen at *runtime* — same binary serves both
    // MHA and GQA-N as long as num_qpkv divides kBlockM). kBlockM is the only
    // structural compile-time knob; pick the tier by max_q after multiplying
    // by num_qpkv in select_config.
    prefill_d128,     // kBlockM=256, 8 warps, 32x32 mfma
    decode_d128_m128, // kBlockM=128, 4 warps, 32x32 mfma
    decode_d128_m32,  // kBlockM=32,  1 warp,  32x32 mfma  (tiny-decode policy)
    decode_d128_m16,  // kBlockM=16,  1 warp,  16x16 mfma  (tiny-decode policy)

    // d=64.
    prefill_d64,      // kBlockM=256, 8 warps, 32x32 mfma
    decode_d64_m128,  // kBlockM=128, 4 warps, 32x32 mfma
    decode_d64_m64,   // kBlockM=64,  2 warps, 32x32 mfma  (decode policy)
    decode_d64_m16,   // kBlockM=16,  1 warp,  16x16 mfma  (tiny-decode policy)
};

// -----------------------------------------------------------------------------
// Per-DataType problem element types.
// -----------------------------------------------------------------------------
template <unified_attention_args::data_type_enum DataType>
struct unified_attention_problem_traits;

template <>
struct unified_attention_problem_traits<unified_attention_args::data_type_enum::fp16>
{
    using qkvp_dtype = ck_tile::half_t;
    using acc_dtype  = float;
    using o_dtype    = ck_tile::half_t;
    using lse_dtype  = float;
};

template <>
struct unified_attention_problem_traits<unified_attention_args::data_type_enum::bf16>
{
    using qkvp_dtype = ck_tile::bf16_t;
    using acc_dtype  = float;
    using o_dtype    = ck_tile::bf16_t;
    using lse_dtype  = float;
};

// FP8 path: Q, K, V are stored / consumed by MFMA as FP8 (e4m3*), but the
// softmax probability P that feeds the second GEMM is *also* FP8 so we can
// reuse the FP8-FP8-F32 MFMA family for PV. The o_dtype stays bf16 (Triton
// reference's output dtype is bf16). Accumulators and LSE remain fp32. The
// type alias resolves to e4m3fnuz on gfx942 and e4m3fn on gfx950 — see
// ck_tile/core/numeric/float8.hpp for the CK_TILE_USE_OCP_FP8 selector.
template <>
struct unified_attention_problem_traits<unified_attention_args::data_type_enum::fp8>
{
    using qkvp_dtype = ck_tile::fp8_t;
    using acc_dtype  = float;
    using o_dtype    = ck_tile::bf16_t;
    using lse_dtype  = float;
};

// =============================================================================
// variant_config<V>
//
// One specialization per KernelVariant. Each exposes the static knobs that
// distinguish that variant from the others:
//
//   HeadSize        : head dimension (compile-time)
//   BlockM          : Q-tile size along the M (token) axis
//   BlockSize       : kBlockN — KV-tile size along the N axis
//   BlockWarps      : warp layout, sequence<M, N, K>
//   WarpGemmShape   : MFMA tile shape, sequence<M, N, K>
//   Pipeline<P>     : pipeline template (default vs decode vs tiny-decode policy)
//   kUseDecodeGrid  : selects 2D-by-seq grid (true) vs Q-block grid (false)
//
// num_queries_per_kv is *not* a compile-time knob: kBlockQ = kBlockM /
// num_qpkv is computed at runtime inside the kernel and pipeline. The only
// constraint is `kBlockM % num_qpkv == 0` (host-side select_config makes sure
// of this).
// =============================================================================
template <KernelVariant V>
struct variant_config;

// Each variant_config exposes `Pipeline<Problem, PageSize>` so the traits
// can pin the page size at compile time. PageSize=0 means "runtime page
// size" (legacy behaviour); the host dispatcher selects a non-zero value
// when it can prove the runtime `args.page_blk_size` matches one of the
// instances we compiled.
template <>
struct variant_config<KernelVariant::prefill_d128>
{
    static constexpr index_t HeadSize  = 128;
    static constexpr index_t BlockM    = 256;
    // KV tile (kBlockN). fp8 runs at 64: the larger tile amortizes the
    // per-tile fixed costs (block barriers, K/V DRAM->LDS latency, address
    // calc, softmax overhead) over 2x the keys, measured +7-12% across
    // sq=2k..75600 at hq=hk=5 d128 (MFMA stays matrix-bound, memwait/barrier
    // per-key roughly halve). 128 falls off an LDS/occupancy cliff (KV double
    // buffer blows the budget for this 1-WG/CU LDS-bound kernel). bf16/fp16
    // auto-halve back to 32 via kBf16HalveBlockN below (2-byte element would
    // double LDS), so this bump is fp8-only by construction.
    //
    // kv128 was historically a ~10x cliff: the 256-VGPR/wave ceiling, hit by the
    // O-accumulator (kBlockM*kHeadDim) plus the DOUBLED score/P tile
    // (kBlockM*kBlockN), spilled hundreds of values to scratch. Two changes remove
    // that pressure so kv128 is now the fastest fp8 prefill tile:
    //   * single-sp (auto-enabled for kPageBlockSize>=128 in the pipeline):
    //     single-buffers the score/P tile -> removes ~122 of the spills.
    //   * cooperative K/V load (default policy): all 8 waves share each tile's
    //     load so no wave owns a 1/4 shard -> removes the remaining ~4 spills.
    // Net: kv128 fp8 prefill is 0-spill / 253-VGPR and ~+15% over the old kv64
    // tile at the canonical b1/sq75600/hq=hk=5/d128 shape.
    //
    // UA_PREFILL_D128_BLOCKSIZE: compile-time KV-tile override (set to 64 to
    // restore the legacy tile for A/B).
#ifndef UA_PREFILL_D128_BLOCKSIZE
#define UA_PREFILL_D128_BLOCKSIZE 128
#endif
    static constexpr index_t BlockSize = UA_PREFILL_D128_BLOCKSIZE;
    using BlockWarps                   = sequence<8, 1, 1>;
    using WarpGemmShape                = sequence<32, 32, 16>;
    template <typename Problem, index_t PageSize = 0, bool IsPaged = true>
    using Pipeline =
        UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineDefaultPolicy, PageSize, IsPaged>;
    static constexpr bool kUseDecodeGrid = false;
};

template <>
struct variant_config<KernelVariant::decode_d128_m128>
{
    static constexpr index_t HeadSize  = 128;
    static constexpr index_t BlockM    = 128;
    static constexpr index_t BlockSize = 32;
    using BlockWarps                   = sequence<4, 1, 1>;
    using WarpGemmShape                = sequence<32, 32, 16>;
    template <typename Problem, index_t PageSize = 0, bool IsPaged = true>
    using Pipeline =
        UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineDefaultPolicy, PageSize, IsPaged>;
    static constexpr bool kUseDecodeGrid = false;
};

template <>
struct variant_config<KernelVariant::decode_d128_m32>
{
    static constexpr index_t HeadSize  = 128;
    static constexpr index_t BlockM    = 32;
    static constexpr index_t BlockSize = 32;
    using BlockWarps                   = sequence<1, 1, 1>;
    using WarpGemmShape                = sequence<32, 32, 16>;
    template <typename Problem, index_t PageSize = 0, bool IsPaged = true>
    using Pipeline =
        UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineTinyDecodePolicy, PageSize, IsPaged>;
    static constexpr bool kUseDecodeGrid = true;
};

template <>
struct variant_config<KernelVariant::decode_d128_m16>
{
    static constexpr index_t HeadSize  = 128;
    static constexpr index_t BlockM    = 16;
    static constexpr index_t BlockSize = 32;
    using BlockWarps                   = sequence<1, 1, 1>;
    using WarpGemmShape                = sequence<16, 16, 32>;
    template <typename Problem, index_t PageSize = 0, bool IsPaged = true>
    using Pipeline =
        UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineTinyDecodePolicy, PageSize, IsPaged>;
    static constexpr bool kUseDecodeGrid = true;
};

template <>
struct variant_config<KernelVariant::prefill_d64>
{
    static constexpr index_t HeadSize  = 64;
    static constexpr index_t BlockM    = 256;
    static constexpr index_t BlockSize = 64;
    using BlockWarps                   = sequence<8, 1, 1>;
    using WarpGemmShape                = sequence<32, 32, 16>;
    template <typename Problem, index_t PageSize = 0, bool IsPaged = true>
    using Pipeline =
        UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineDefaultPolicy, PageSize, IsPaged>;
    static constexpr bool kUseDecodeGrid = false;
};

template <>
struct variant_config<KernelVariant::decode_d64_m128>
{
    static constexpr index_t HeadSize  = 64;
    static constexpr index_t BlockM    = 128;
    static constexpr index_t BlockSize = 64;
    using BlockWarps                   = sequence<4, 1, 1>;
    using WarpGemmShape                = sequence<32, 32, 16>;
    template <typename Problem, index_t PageSize = 0, bool IsPaged = true>
    using Pipeline =
        UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineDefaultPolicy, PageSize, IsPaged>;
    static constexpr bool kUseDecodeGrid = false;
};

template <>
struct variant_config<KernelVariant::decode_d64_m64>
{
    static constexpr index_t HeadSize  = 64;
    static constexpr index_t BlockM    = 64;
    static constexpr index_t BlockSize = 64;
    using BlockWarps                   = sequence<2, 1, 1>;
    using WarpGemmShape                = sequence<32, 32, 16>;
    template <typename Problem, index_t PageSize = 0, bool IsPaged = true>
    using Pipeline =
        UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineDecodePolicy, PageSize, IsPaged>;
    static constexpr bool kUseDecodeGrid = true;
};

template <>
struct variant_config<KernelVariant::decode_d64_m16>
{
    static constexpr index_t HeadSize  = 64;
    static constexpr index_t BlockM    = 16;
    static constexpr index_t BlockSize = 64;
    using BlockWarps                   = sequence<1, 1, 1>;
    using WarpGemmShape                = sequence<16, 16, 32>;
    template <typename Problem, index_t PageSize = 0, bool IsPaged = true>
    using Pipeline =
        UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineTinyDecodePolicy, PageSize, IsPaged>;
    static constexpr bool kUseDecodeGrid = true;
};

// =============================================================================
// unified_attention_kernel_traits<V, DataType, IsMasking>
//
// Single templated trait. Pulls per-variant knobs from variant_config<V> and
// per-dtype element types from unified_attention_problem_traits<DataType>.
// =============================================================================
// kPageSize: optional compile-time pin of the runtime `page_size`. Default
// 0 keeps the legacy runtime-page-size behaviour; a non-zero value lets the
// pipeline strength-reduce the per-tile arithmetic *and* widen the Tier 0 /
// Tier 2 gate from the conservative `KY0_step_N <= 16` hedge to the real
// `KY0_step_N <= kPageSize` condition. The host dispatcher (dispatch_variant
// in unified_attention.cpp) picks the matching instance at launch time.
template <KernelVariant V,
          unified_attention_args::data_type_enum DataType,
          bool IsMasking,
          ck_tile::index_t kPageSize_ = 0,
          bool IsLocal_               = false,
          bool IsPaged_               = true>
struct unified_attention_kernel_traits
{
    using cfg = variant_config<V>;
    using dt  = unified_attention_problem_traits<DataType>;

    static constexpr auto             date_type  = DataType;
    static constexpr bool             is_masking = IsMasking;
    static constexpr bool             is_local   = IsLocal_;
    static constexpr KernelVariant    variant    = V;
    static constexpr ck_tile::index_t kPageSize  = kPageSize_;

    static constexpr index_t HEAD_SIZE      = cfg::HeadSize;
    static constexpr index_t kBlockM        = cfg::BlockM;
    static constexpr bool    kUseDecodeGrid = cfg::kUseDecodeGrid;

    // bf16/fp16 carry a 2-byte element vs fp8's 1 byte, so at the same kBlockN
    // they double both LDS usage and per-tile VGPR pressure. The decode probe
    // (ua-test-scripts/probe_decode_d128.sh) showed bf16 saturating VGPR=256
    // with AGPR overflow (44-106 AGPRs) and ~2x the LDS of fp8 on the m16
    // tier — the LDS pressure alone caps decode_d128_m16 at 1 CTA/CU. We
    // halve kBlockN for bf16/fp16 to shed LDS and VGPR pressure, trading a
    // small per-iter overhead for a big occupancy boost.
    //
    // The halved kBlockN must satisfy both gemm constraints:
    //   QK gemm: kBlockN is the N axis  -> kBlockN >= WarpGemm::N
    //   PV gemm: kBlockN is the K axis  -> kBlockN >= WarpGemm::K
    // When `cfg::BlockSize/2 < WarpGemm::K` we additionally swap WarpGemm::K
    // to the halved kBlockN so the smaller-K MFMA is used. For our variants
    // this only hits decode_d128_m16 (WG=<16,16,32> -> <16,16,16>); the d=64
    // tiers already fit the smaller tile under the same 16x16x32 / 32x32x16
    // MFMA. The 32x32 N-warps (m32/m128/prefill) cannot drop their kBlockN
    // below 32 (WG::N=32) and stay at the un-halved size.
    static constexpr index_t WGM_ = cfg::WarpGemmShape::at(number<0>{});
    static constexpr index_t WGN_ = cfg::WarpGemmShape::at(number<1>{});
    static constexpr index_t WGK_ = cfg::WarpGemmShape::at(number<2>{});
    static constexpr bool kBf16HalveBlockN =
        (DataType != unified_attention_args::data_type_enum::fp8) &&
        (cfg::BlockSize / 2 >= WGN_);
    static constexpr index_t BLOCK_SIZE =
        kBf16HalveBlockN ? cfg::BlockSize / 2 : cfg::BlockSize;
    // Swap WarpGemm::K down to BLOCK_SIZE when the halved kBlockN dropped
    // below the original WarpGemm::K. PVAttrNumAccess in GetPVBlockGemm
    // recomputes from the new WarpGemm shape (lanes_in_K * SubMinDim rule)
    // so the smaller-K MFMA tiles cleanly.
    // fp8 d128 prefill: use the wide CDNA4 MFMA (v_mfma_f32_32x32x64_f8f6f4)
    // instead of the narrow 32x32x16 — 4x fewer MFMA instructions for the same
    // FLOPs (the QK contraction over kHeadDim=128 drops from 8 to 2 k-steps).
    // Gated to fp8/d128/prefill where the contraction tiles (kHeadDim for QK,
    // BLOCK_SIZE for PV) are multiples of 64; other dtypes have no 32x32x64
    // MFMA and keep the narrow tile. The barrier-free QK-C->PV-A relayout for
    // this tile lives in fmha_alu1 (strategy C).
    static constexpr bool kFp8WideMma =
        (UA_FP8_WIDE_MMA != 0) &&
        (DataType == unified_attention_args::data_type_enum::fp8) &&
        (HEAD_SIZE == 128) && (V == KernelVariant::prefill_d128) &&
        (BLOCK_SIZE % 64 == 0);
    using unified_attention_warp_gemm_shape = std::conditional_t<
        kFp8WideMma,
        sequence<WGM_, WGN_, 64>,
        std::conditional_t<(kBf16HalveBlockN && BLOCK_SIZE < WGK_),
                           sequence<WGM_, WGN_, BLOCK_SIZE>,
                           typename cfg::WarpGemmShape>>;

    // The 2nd entry of the BlockTile is the static `kBlockQ` exposed via
    // `UnifiedAttentionShape::kBlockQ`. Now that the kernel always reads
    // kBlockQ from `args.num_queries_per_kv` at runtime, this static value
    // is only the fallback when no num_qpkv was plumbed through (which never
    // happens in practice). Anchor it at kBlockM so the static "looks like
    // num_qpkv == 1" and any (kBlockM, num_qpkv) such that kBlockM % num_qpkv
    // == 0 works without touching this trait.
    using unified_attention_block_tile  = sequence<kBlockM, kBlockM, BLOCK_SIZE, HEAD_SIZE>;
    using unified_attention_block_warps = typename cfg::BlockWarps;

    using unified_attention_shape = TileUnifiedAttentionShape<unified_attention_block_tile,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              true>; // IsVLayoutRowMajor

    using unified_attention_traits = TileUnifiedAttentionTraits<true,  // kPadSeqLenQ_
                                                                false, // kPadHeadDimQ
                                                                -1>;   // kBlockPerCu
    using unified_attention_mask   = GenericAttentionMask<IsMasking, IsLocal_>;

    using unified_attention_pipeline_problem =
        UnifiedAttentionPipelineProblem<typename dt::qkvp_dtype,
                                        typename dt::qkvp_dtype,
                                        typename dt::qkvp_dtype,
                                        typename dt::acc_dtype,
                                        typename dt::acc_dtype,
                                        typename dt::acc_dtype,
                                        typename dt::lse_dtype,
                                        typename dt::qkvp_dtype,
                                        typename dt::acc_dtype,
                                        typename dt::o_dtype,
                                        unified_attention_shape,
                                        unified_attention_mask,
                                        unified_attention_traits>;

    using unified_attention_pipeline =
        typename cfg::template Pipeline<unified_attention_pipeline_problem, kPageSize_, IsPaged_>;

    using epilogue =
        Default2DEpilogue<Default2DEpilogueProblem<typename dt::acc_dtype,
                                                   typename dt::o_dtype,
                                                   true, // kPadM
                                                   true, // kPadN
                                                   true  // UseRawStore
                                                   >>;

    using kernel = UnifiedAttentionKernel<unified_attention_pipeline, epilogue>;
};

// =============================================================================
// Kernel launch — common helper. Picks the grid layout from
// Traits::kUseDecodeGrid; all other launch args are identical across variants.
// =============================================================================
template <typename Kernel, bool UseDecodeGrid = false>
float unified_attention_kernel_launch(const unified_attention_args& args,
                                      const stream_config& config)
{
    // kBlockQ is derived from the runtime num_queries_per_kv now -- the
    // static `Kernel::kBlockQ` is anchored at kBlockM and would over-count
    // tiles for GQA workloads. We assert kBlockM % num_qpkv == 0 in
    // select_config so this integer divide is always exact.
    const index_t kBlockQ            = Kernel::kBlockM / args.num_queries_per_kv;
    const index_t total_num_q_blocks = args.num_tokens / kBlockQ + args.num_seqs;
    auto kargs                 = Kernel::MakeKargs(args.q_ptr,
                                   args.k_ptr,
                                   args.v_ptr,
                                   args.o_ptr,
                                   args.num_blks,
                                   args.num_head_q,
                                   args.num_queries_per_kv,
                                   args.scale_s,
                                   args.scale,
                                   args.scale_k,
                                   args.scale_v,
                                   args.scale_out,
                                   args.q_descale,
                                   args.k_descale,
                                   args.v_descale,
                                   args.page_blk_size,
                                   total_num_q_blocks,
                                   args.query_stride_0,
                                   args.query_stride_1,
                                   args.stride_k_cache_0,
                                   args.stride_k_cache_1,
                                   args.stride_k_cache_2,
                                   args.stride_k_cache_3,
                                   args.stride_v_cache_0,
                                   args.stride_v_cache_1,
                                   args.stride_v_cache_2,
                                   args.stride_v_cache_3,
                                   args.output_stride_0,
                                   args.output_stride_1,
                                   args.block_tables_ptr,
                                   args.block_table_stride,
                                   args.seq_lens_ptr,
                                   args.query_start_len_ptr,
                                   args.num_seqs,
                                   args.num_splits,
                                   args.lse_acc_ptr,
                                   args.o_acc_ptr,
                                   args.split_stride_lse_acc,
                                   args.split_stride_o_acc,
                                   args.nhead_stride_lse_acc,
                                   args.nhead_stride_o_acc,
                                   args.cache_ptr_int32_overflow_possible,
                                   args.window_size_left,
                                   args.window_size_right,
                                   args.is_top_left,
                                   args.kv_start_len_ptr);

    dim3 grids;
    if constexpr(UseDecodeGrid)
    {
        grids = Kernel::GridSizeDecode(args.num_head_q / args.num_queries_per_kv,
                                       args.num_seqs,
                                       args.num_splits);
    }
    else
    {
        grids = Kernel::GridSize2D(args.num_head_q / args.num_queries_per_kv,
                                   total_num_q_blocks,
                                   args.num_splits);
    }
    constexpr dim3 blocks         = Kernel::BlockSize();
    constexpr index_t kBlockPerCu = Kernel::kBlockPerCu;

    return launch_kernel(config, make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

// =============================================================================
// Per-instance dispatch. Each instance .cpp specializes this for its
// (V, DataType, IsMasking) tuple via INST_UNIFIED_ATTENTION_DISPATCH.
//
// Return: (launched?, elapsed_ms). elapsed_ms is valid only when launched.
// =============================================================================
template <typename Traits>
std::pair<bool, float> unified_attention_kernel_dispatch(const unified_attention_args& args,
                                                         const stream_config& config);

} // namespace ck_tile

// Profiling slim-build hook. When a TU is compiled with -DUA_STUB_INSTANCE
// (injected per-source by the AITER_UA_TRACE_INSTANCES build knob), the
// instance macros below emit a trivial host stub that returns {false,-1.f}
// and instantiate NO device kernel. The symbol still exists (so the runtime
// dispatch switch links), but the kernel is absent from the code object --
// which is what keeps rocprofv3 ATT disassembly fast when only one instance
// is wanted. UA_KERNEL_DISPATCH_RESULT picks the real launch vs the stub.
#ifdef UA_STUB_INSTANCE
#define UA_KERNEL_DISPATCH_RESULT(TRAITS_) (std::make_pair(false, -1.f))
#else
#define UA_KERNEL_DISPATCH_RESULT(TRAITS_)                                                   \
    std::make_pair(true,                                                                     \
                   unified_attention_kernel_launch<typename TRAITS_::kernel,                 \
                                                   TRAITS_::kUseDecodeGrid>(args, config))
#endif

// One-line instantiation per (V, DataType, IsMasking, PageSize, IsLocal)
// combination. Each instance .cpp consists of exactly one of these calls.
// PAGE_SIZE_ = 0 is the legacy runtime-page-size instance (catch-all
// fallback). IS_LOCAL_ = false is the non-SWA path (causal / no-mask);
// IS_LOCAL_ = true compiles the SWA-capable kernel that honours both the
// left and right window bounds inside the mask (used from Phase 3 on).
#define INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(VARIANT_, DTYPE_, IS_MASK_, PAGE_SIZE_,     \
                                                 IS_LOCAL_)                                  \
    template <>                                                                              \
    std::pair<bool, float> unified_attention_kernel_dispatch<                                \
        unified_attention_kernel_traits<KernelVariant::VARIANT_,                             \
                                        unified_attention_args::data_type_enum::DTYPE_,      \
                                        IS_MASK_,                                            \
                                        PAGE_SIZE_,                                          \
                                        IS_LOCAL_>>([[maybe_unused]] const unified_attention_args& args, \
                                                    [[maybe_unused]] const stream_config& config) \
    {                                                                                        \
        using Traits [[maybe_unused]] = unified_attention_kernel_traits<                     \
            KernelVariant::VARIANT_,                                                         \
            unified_attention_args::data_type_enum::DTYPE_,                                  \
            IS_MASK_,                                                                        \
            PAGE_SIZE_,                                                                      \
            IS_LOCAL_>;                                                                      \
        return UA_KERNEL_DISPATCH_RESULT(Traits);                                            \
    }

// Backward-compat wrappers — every existing instance .cpp uses one of these
// and defaults to `IsLocal = false` (the non-SWA path).
#define INST_UNIFIED_ATTENTION_DISPATCH_PS(VARIANT_, DTYPE_, IS_MASK_, PAGE_SIZE_)            \
    INST_UNIFIED_ATTENTION_DISPATCH_PS_LOCAL(VARIANT_, DTYPE_, IS_MASK_, PAGE_SIZE_, false)

#define INST_UNIFIED_ATTENTION_DISPATCH(VARIANT_, DTYPE_, IS_MASK_)                          \
    INST_UNIFIED_ATTENTION_DISPATCH_PS(VARIANT_, DTYPE_, IS_MASK_, 0)

// Contiguous (THD) KV instance — IsPaged = false, runtime page size (0),
// non-SWA. Mirrors INST_UNIFIED_ATTENTION_DISPATCH but flips the IsPaged_
// trait so the pipeline compiles out all block_tables / page math.
#define INST_UNIFIED_ATTENTION_DISPATCH_NONPAGED(VARIANT_, DTYPE_, IS_MASK_)                  \
    template <>                                                                              \
    std::pair<bool, float> unified_attention_kernel_dispatch<                                \
        unified_attention_kernel_traits<KernelVariant::VARIANT_,                             \
                                        unified_attention_args::data_type_enum::DTYPE_,      \
                                        IS_MASK_,                                            \
                                        /*kPageSize=*/0,                                     \
                                        /*IsLocal=*/false,                                   \
                                        /*IsPaged=*/false>>([[maybe_unused]] const unified_attention_args& args, \
                                                            [[maybe_unused]] const stream_config& config) \
    {                                                                                        \
        using Traits [[maybe_unused]] = unified_attention_kernel_traits<                     \
            KernelVariant::VARIANT_,                                                         \
            unified_attention_args::data_type_enum::DTYPE_,                                  \
            IS_MASK_,                                                                        \
            /*kPageSize=*/0,                                                                 \
            /*IsLocal=*/false,                                                               \
            /*IsPaged=*/false>;                                                              \
        return UA_KERNEL_DISPATCH_RESULT(Traits);                                            \
    }
