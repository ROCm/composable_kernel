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

template <unified_attention_args::data_type_enum DataType, bool IsMasking>
struct unified_attention_kernel_traits
{
    static constexpr auto date_type  = DataType;
    static constexpr bool is_masking = IsMasking;

    static constexpr index_t BLOCK_M    = 256;
    static constexpr index_t BLOCK_SIZE = 32;
    static constexpr index_t HEAD_SIZE  = 128;

    // TODO please fix this to support also other num_queries_per_kv
    static constexpr index_t num_queries_per_kv = 1;
    static constexpr index_t BLOCK_Q            = BLOCK_M / num_queries_per_kv;

    //                                    BLOCK_M BLOCK_Q   BLOCK_SIZE  HEAD_SIZE
    using unified_attention_block_tile      = sequence<BLOCK_M, BLOCK_Q, BLOCK_SIZE, HEAD_SIZE>;
    using unified_attention_warp_gemm_shape = sequence<32, 32, 16>;
    // need to have 8 warps per workgroup to have warp specialization
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

template <typename Kernel>
float unified_attention_kernel_launch(const unified_attention_args& args,
                                      const stream_config& config)
{
    index_t BLOCK_Q = Kernel::BLOCK_Q;
    assert(args.num_queries_per_kv == Kernel::num_queries_per_kv &&
           "argument num_queries_per_kv must equal compiled num_queries_per_kv");
    assert(args.BLOCK_SIZE == Kernel::BLOCK_SIZE &&
           "argument BLOCK_SIZE must equal compiled BLOCK_SIZE");
    assert(BLOCK_Q == BLOCK_M / args.num_queries_per_kv &&
           "BLOCK_Q must equal BLOCK_M / num_queries_per_kv");
    index_t total_num_q_blocks = args.num_tokens / BLOCK_Q + args.num_seqs;
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
