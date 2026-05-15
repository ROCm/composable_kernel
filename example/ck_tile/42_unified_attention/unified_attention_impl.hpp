// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <utility>

#include "ck_tile/core/numeric/bfloat16.hpp"
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

template <>
struct variant_config<KernelVariant::prefill_d128>
{
    static constexpr index_t HeadSize  = 128;
    static constexpr index_t BlockM    = 256;
    static constexpr index_t BlockSize = 32;
    using BlockWarps                   = sequence<8, 1, 1>;
    using WarpGemmShape                = sequence<32, 32, 16>;
    template <typename Problem>
    using Pipeline                       = UnifiedAttentionPipeline<Problem>;
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
    template <typename Problem>
    using Pipeline                       = UnifiedAttentionPipeline<Problem>;
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
    template <typename Problem>
    using Pipeline = UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineTinyDecodePolicy>;
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
    template <typename Problem>
    using Pipeline = UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineTinyDecodePolicy>;
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
    template <typename Problem>
    using Pipeline                       = UnifiedAttentionPipeline<Problem>;
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
    template <typename Problem>
    using Pipeline                       = UnifiedAttentionPipeline<Problem>;
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
    template <typename Problem>
    using Pipeline = UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineDecodePolicy>;
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
    template <typename Problem>
    using Pipeline = UnifiedAttentionPipeline<Problem, UnifiedAttentionPipelineTinyDecodePolicy>;
    static constexpr bool kUseDecodeGrid = true;
};

// =============================================================================
// unified_attention_kernel_traits<V, DataType, IsMasking>
//
// Single templated trait. Pulls per-variant knobs from variant_config<V> and
// per-dtype element types from unified_attention_problem_traits<DataType>.
// =============================================================================
template <KernelVariant V,
          unified_attention_args::data_type_enum DataType,
          bool IsMasking>
struct unified_attention_kernel_traits
{
    using cfg = variant_config<V>;
    using dt  = unified_attention_problem_traits<DataType>;

    static constexpr auto          date_type  = DataType;
    static constexpr bool          is_masking = IsMasking;
    static constexpr KernelVariant variant    = V;

    static constexpr index_t HEAD_SIZE      = cfg::HeadSize;
    static constexpr index_t kBlockM        = cfg::BlockM;
    static constexpr index_t BLOCK_SIZE     = cfg::BlockSize;
    static constexpr bool    kUseDecodeGrid = cfg::kUseDecodeGrid;

    // The 2nd entry of the BlockTile is the static `kBlockQ` exposed via
    // `UnifiedAttentionShape::kBlockQ`. Now that the kernel always reads
    // kBlockQ from `args.num_queries_per_kv` at runtime, this static value
    // is only the fallback when no num_qpkv was plumbed through (which never
    // happens in practice). Anchor it at kBlockM so the static "looks like
    // num_qpkv == 1" and any (kBlockM, num_qpkv) such that kBlockM % num_qpkv
    // == 0 works without touching this trait.
    using unified_attention_block_tile      = sequence<kBlockM, kBlockM, BLOCK_SIZE, HEAD_SIZE>;
    using unified_attention_warp_gemm_shape = typename cfg::WarpGemmShape;
    using unified_attention_block_warps     = typename cfg::BlockWarps;

    using unified_attention_shape = TileUnifiedAttentionShape<unified_attention_block_tile,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              true>; // IsVLayoutRowMajor

    using unified_attention_traits = TileUnifiedAttentionTraits<true,  // kPadSeqLenQ_
                                                                false, // kPadHeadDimQ
                                                                -1>;   // kBlockPerCu
    using unified_attention_mask   = GenericAttentionMask<IsMasking, /*IsLocal=*/false>;

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
        typename cfg::template Pipeline<unified_attention_pipeline_problem>;

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
                                   args.cache_ptr_int32_overflow_possible);

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

// One-line instantiation per (V, DataType, IsMasking) combination. Each
// instance .cpp consists of exactly one of these calls.
#define INST_UNIFIED_ATTENTION_DISPATCH(VARIANT_, DTYPE_, IS_MASK_)                          \
    template <>                                                                              \
    std::pair<bool, float> unified_attention_kernel_dispatch<                                \
        unified_attention_kernel_traits<KernelVariant::VARIANT_,                             \
                                        unified_attention_args::data_type_enum::DTYPE_,      \
                                        IS_MASK_>>(const unified_attention_args& args,       \
                                                   const stream_config& config)              \
    {                                                                                        \
        using Traits = unified_attention_kernel_traits<                                      \
            KernelVariant::VARIANT_,                                                         \
            unified_attention_args::data_type_enum::DTYPE_,                                  \
            IS_MASK_>;                                                                       \
        return std::make_pair(true,                                                          \
            unified_attention_kernel_launch<typename Traits::kernel,                         \
                                            Traits::kUseDecodeGrid>(args, config));          \
    }
