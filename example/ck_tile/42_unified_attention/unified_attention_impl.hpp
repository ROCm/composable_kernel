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

#define INST_UNIFIED_ATTENTION_DISPATCH(kernel_traits)                                   \
    template <>                                                                          \
    std::pair<bool, float> unified_attention_kernel_dispatch<kernel_traits>(             \
        const unified_attention_args& args, const stream_config& config)                 \
    {                                                                                    \
        return std::make_pair(                                                           \
            true, unified_attention_kernel_launch<kernel_traits::kernel>(args, config)); \
    }

namespace ck_tile {

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

// Parameterized kernel traits: DataType, IsMasking, HeadSize, BlockM, NumQueriesPerKV
template <unified_attention_args::data_type_enum DataType,
          bool IsMasking,
          index_t HeadSize_     = 128,
          index_t BlockM_       = 256,
          index_t NumQPerKV_    = 1>
struct unified_attention_kernel_traits
{
    static constexpr auto date_type  = DataType;
    static constexpr bool is_masking = IsMasking;

    static constexpr index_t kBlockM    = BlockM_;
    static constexpr index_t HEAD_SIZE  = HeadSize_;
    // On gfx950 with 128-bit loads (KVector=8), NumIssues = kPageBlockSize*HeadSize/4096.
    // For HeadSize<=64 we need kPageBlockSize>=64 to keep NumIssues>=1.
    static constexpr index_t BLOCK_SIZE = (HEAD_SIZE <= 64) ? 64 : 32;

    static constexpr index_t num_queries_per_kv = NumQPerKV_;
    static constexpr index_t kBlockQ            = kBlockM / num_queries_per_kv;

    //                                    kBlockM kBlockQ   BLOCK_SIZE  HEAD_SIZE
    using unified_attention_block_tile      = sequence<kBlockM, kBlockQ, BLOCK_SIZE, HEAD_SIZE>;

    using unified_attention_warp_gemm_shape = sequence<32, 32, 16>;
    // 8 warps for warp specialization; kBlockM must be 8 * 32 = 256
    using unified_attention_block_warps = sequence<8, 1, 1>;

    using unified_attention_shape = TileUnifiedAttentionShape<unified_attention_block_tile,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              true // IsVLayoutRowMajor
                                                              >;

    using unified_attention_traits = TileUnifiedAttentionTraits<true,  // kPadSeqLenQ_
                                                                false, // kPadHeadDimQ
                                                                -1     // kBlockPerCu
                                                                >;

    using unified_attention_mask = GenericAttentionMask<IsMasking, /*IsLocal=*/false>;

    using unified_attention_pipeline_problem = UnifiedAttentionPipelineProblem<
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::lse_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::o_dtype,
        unified_attention_shape,
        unified_attention_mask,
        unified_attention_traits>;

    using unified_attention_pipeline = UnifiedAttentionPipeline<unified_attention_pipeline_problem>;

    using epilogue = Default2DEpilogue<
        Default2DEpilogueProblem<typename unified_attention_problem_traits<date_type>::acc_dtype,
                                 typename unified_attention_problem_traits<date_type>::o_dtype,
                                 true, // kPadM
                                 true, // kPadM
                                 true  // UseRawStore
                                 >>;

    using kernel = UnifiedAttentionKernel<unified_attention_pipeline, epilogue>;
};

// Decode-tuned traits: 4 warps (1 warp group), kBlockM=128, serial pipeline.
// Uses the single-warp-group path in UnifiedAttentionPipeline.
template <unified_attention_args::data_type_enum DataType,
          bool IsMasking,
          index_t HeadSize_  = 128,
          index_t BlockM_    = 128,
          index_t NumQPerKV_ = 1>
struct unified_attention_decode_kernel_traits
{
    static constexpr auto date_type  = DataType;
    static constexpr bool is_masking = IsMasking;

    static constexpr index_t kBlockM    = BlockM_;
    static constexpr index_t HEAD_SIZE  = HeadSize_;
    static constexpr index_t BLOCK_SIZE = (HEAD_SIZE <= 64) ? 64 : 32;

    static constexpr index_t num_queries_per_kv = NumQPerKV_;
    static constexpr index_t kBlockQ            = kBlockM / num_queries_per_kv;

    //                                    kBlockM kBlockQ   BLOCK_SIZE  HEAD_SIZE
    using unified_attention_block_tile      = sequence<kBlockM, kBlockQ, BLOCK_SIZE, HEAD_SIZE>;
    using unified_attention_warp_gemm_shape = sequence<32, 32, 16>;
    // 4 warps -> kBlockSize = 256 threads -> NumWarpGroups = 1
    using unified_attention_block_warps     = sequence<4, 1, 1>;

    using unified_attention_shape = TileUnifiedAttentionShape<unified_attention_block_tile,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              true>;

    using unified_attention_traits = TileUnifiedAttentionTraits<true, false, -1>;
    using unified_attention_mask   = GenericAttentionMask<IsMasking, false>;

    using unified_attention_pipeline_problem = UnifiedAttentionPipelineProblem<
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::lse_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::o_dtype,
        unified_attention_shape,
        unified_attention_mask,
        unified_attention_traits>;

    using unified_attention_pipeline = UnifiedAttentionPipeline<unified_attention_pipeline_problem>;

    using epilogue = Default2DEpilogue<
        Default2DEpilogueProblem<typename unified_attention_problem_traits<date_type>::acc_dtype,
                                 typename unified_attention_problem_traits<date_type>::o_dtype,
                                 true, true, true>>;

    using kernel = UnifiedAttentionKernel<unified_attention_pipeline, epilogue>;
};

// Small decode traits: 2 warps, kBlockM=64, decode policy (NumWarpPerGroup=2).
// Uses 1D warp layout (sequence<2,1,1>) so no softmax reduction changes needed.
template <unified_attention_args::data_type_enum DataType,
          bool IsMasking,
          index_t HeadSize_  = 64,
          index_t BlockM_    = 64,
          index_t NumQPerKV_ = 8>
struct unified_attention_decode_small_kernel_traits
{
    static constexpr auto date_type  = DataType;
    static constexpr bool is_masking = IsMasking;

    static constexpr index_t kBlockM    = BlockM_;
    static constexpr index_t HEAD_SIZE  = HeadSize_;
    static constexpr index_t BLOCK_SIZE = (HEAD_SIZE <= 64) ? 64 : 32;

    static constexpr index_t num_queries_per_kv = NumQPerKV_;
    static constexpr index_t kBlockQ            = kBlockM / num_queries_per_kv;

    using unified_attention_block_tile      = sequence<kBlockM, kBlockQ, BLOCK_SIZE, HEAD_SIZE>;
    using unified_attention_warp_gemm_shape = sequence<32, 32, 16>;
    // 2 warps along M: kBlockM=2*32=64, kBlockSize=128, NumWarpGroups=1
    using unified_attention_block_warps     = sequence<2, 1, 1>;

    using unified_attention_shape = TileUnifiedAttentionShape<unified_attention_block_tile,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              unified_attention_block_warps,
                                                              unified_attention_warp_gemm_shape,
                                                              true>;

    using unified_attention_traits = TileUnifiedAttentionTraits<true, false, -1>;
    using unified_attention_mask   = GenericAttentionMask<IsMasking, false>;

    using unified_attention_pipeline_problem = UnifiedAttentionPipelineProblem<
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::lse_dtype,
        typename unified_attention_problem_traits<date_type>::qkvp_dtype,
        typename unified_attention_problem_traits<date_type>::acc_dtype,
        typename unified_attention_problem_traits<date_type>::o_dtype,
        unified_attention_shape,
        unified_attention_mask,
        unified_attention_traits>;

    using unified_attention_pipeline =
        UnifiedAttentionPipeline<unified_attention_pipeline_problem,
                                 UnifiedAttentionPipelineDecodePolicy>;

    using epilogue = Default2DEpilogue<
        Default2DEpilogueProblem<typename unified_attention_problem_traits<date_type>::acc_dtype,
                                 typename unified_attention_problem_traits<date_type>::o_dtype,
                                 true, true, true>>;

    using kernel = UnifiedAttentionKernel<unified_attention_pipeline, epilogue>;
};

template <typename Kernel>
float unified_attention_kernel_launch(const unified_attention_args& args,
                                      const stream_config& config)
{
    constexpr index_t kBlockQ = Kernel::kBlockQ;
    index_t total_num_q_blocks = args.num_tokens / kBlockQ + args.num_seqs;
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
                                   args.num_seqs);

    dim3 grids = Kernel::GridSize2D(args.num_head_q / args.num_queries_per_kv, total_num_q_blocks);
    constexpr dim3 blocks         = Kernel::BlockSize();
    constexpr index_t kBlockPerCu = Kernel::kBlockPerCu;

    return launch_kernel(config, make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

// return value:
//   first  = whether the kernel was launched (true = launched, false = skipped)
//   second = elapsed time (ms) of the kernel launch, valid only if first == true
template <typename KernelTraits>
std::pair<bool, float> unified_attention_kernel_dispatch(const unified_attention_args& args,
                                                         const stream_config& config);

} // namespace ck_tile
