// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <utility>

#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"

#include "fmha_fwd_v3.hpp"
#include "mask.hpp"

#define INST_FMHA_FWD_V3_DISPATCH(kernel_traits)                                               \
    template <>                                                                                \
    std::pair<bool, float> fmha_fwd_v3_kernel_dispatch<kernel_traits>(                         \
        const fmha_fwd_v3_args& args, const stream_config& config)                             \
    {                                                                                          \
        return std::make_pair(true,                                                            \
                              fmha_fwd_v3_kernel_launch<kernel_traits::kernel>(args, config)); \
    }

namespace ck_tile {

template <fmha_fwd_v3_args::data_type_enum DataType>
struct fmha_fwd_v3_problem_traits;

template <>
struct fmha_fwd_v3_problem_traits<fmha_fwd_v3_args::data_type_enum::fp16>
{
    using qkvp_dtype = ck_tile::half_t;
    using acc_dtype  = float;
    using o_dtype    = ck_tile::half_t;
    using lse_dtype  = float;
};

template <>
struct fmha_fwd_v3_problem_traits<fmha_fwd_v3_args::data_type_enum::bf16>
{
    using qkvp_dtype = ck_tile::bf16_t;
    using acc_dtype  = float;
    using o_dtype    = ck_tile::bf16_t;
    using lse_dtype  = float;
};

template <fmha_fwd_v3_args::data_type_enum DataType, bool IsVariableSeqlen, bool IsMasking>
struct fmha_fwd_v3_kernel_traits
{
    static constexpr auto date_type          = DataType;
    static constexpr bool is_variable_seqlen = IsVariableSeqlen;
    static constexpr bool is_masking         = IsMasking;

    //                                    M0   N0  K0   N1   K1
    using fmha_block_tile      = sequence<256, 32, 128, 128, 32, 128>;
    using fmha_warp_gemm_shape = sequence<32, 32, 16>;
    using fmha_block_warps     = sequence<8, 1, 1>;

    using fmha_shape = TileFmhaShape<fmha_block_tile,
                                     fmha_block_warps,
                                     fmha_warp_gemm_shape,
                                     fmha_block_warps,
                                     fmha_warp_gemm_shape,
                                     true // IsVLayoutRowMajor
                                     >;

    using fmha_traits = TileFmhaFwdV3Traits<true,  // kPadSeqLenQ
                                            true,  // kPadSeqLenK
                                            false, // kPadHeadDimQ
                                            false, // kPadHeadDimV
                                            false, // kStoreLSE
                                            -1     // kBlockPerCu
                                            >;

    using fmha_mask = GenericAttentionMask<IsMasking, /*IsLocal=*/false>;

    using fmha_pipeline_problem =
        BlockFmhaFwdV3PipelineProblem<typename fmha_fwd_v3_problem_traits<date_type>::qkvp_dtype,
                                      typename fmha_fwd_v3_problem_traits<date_type>::qkvp_dtype,
                                      typename fmha_fwd_v3_problem_traits<date_type>::qkvp_dtype,
                                      typename fmha_fwd_v3_problem_traits<date_type>::acc_dtype,
                                      typename fmha_fwd_v3_problem_traits<date_type>::acc_dtype,
                                      typename fmha_fwd_v3_problem_traits<date_type>::lse_dtype,
                                      typename fmha_fwd_v3_problem_traits<date_type>::qkvp_dtype,
                                      typename fmha_fwd_v3_problem_traits<date_type>::acc_dtype,
                                      typename fmha_fwd_v3_problem_traits<date_type>::o_dtype,
                                      fmha_shape,
                                      IsVariableSeqlen,
                                      fmha_mask,
                                      fmha_traits>;

    using fmha_pipeline = BlockFmhaFwdV3Pipeline<fmha_pipeline_problem>;

    using epilogue = Default2DEpilogue<
        Default2DEpilogueProblem<typename fmha_fwd_v3_problem_traits<date_type>::acc_dtype,
                                 typename fmha_fwd_v3_problem_traits<date_type>::o_dtype,
                                 true, // kPadM
                                 true, // kPadM
                                 true  // UseRawStore
                                 >>;

    using kernel = FmhaFwdV3Kernel<fmha_pipeline, epilogue>;
};

template <typename Kernel>
float fmha_fwd_v3_kernel_launch(const fmha_fwd_v3_args& args, const stream_config& config)
{
    /// NOTICE: This was borrowed from Aiter. Make sure the selected remap_opt setting truly
    /// maximizes the kernel's performance.
    int remap_opt = 2;
    if(args.mask_type != static_cast<int>(mask_enum::no_mask) &&
       ((args.nhead_q % 8 != 0) || (16384 < args.seqlen_q)))
    {
        if(65536 <= args.seqlen_q)
        {
            remap_opt = 0;
        }
        else
        {
            remap_opt = 1;
        }
    }

    auto kargs = Kernel::MakeKargs(args.q_ptr,
                                   args.k_ptr,
                                   args.v_ptr,
                                   nullptr, // lse_ptr
                                   args.o_ptr,
                                   args.seqlen_q,
                                   args.seqlen_k,
                                   args.hdim_qk,
                                   args.hdim_v,
                                   args.nhead_q,
                                   args.nhead_q / args.nhead_kv,
                                   args.softmax_scale,
                                   args.stride_q,
                                   args.stride_k,
                                   args.stride_v,
                                   args.stride_o,
                                   args.nhead_stride_q,
                                   args.nhead_stride_k,
                                   args.nhead_stride_v,
                                   0, // nhead_stride_lse
                                   args.nhead_stride_o,
                                   args.batch_stride_q,
                                   args.batch_stride_k,
                                   args.batch_stride_v,
                                   0, // batch_stride_lse
                                   args.batch_stride_o,
                                   args.window_size_left,
                                   args.window_size_right,
                                   args.mask_type,
                                   remap_opt,
                                   args.cu_seqlen_q_ptr,
                                   args.cu_seqlen_kv_ptr);

    dim3 grids            = Kernel::GridSize(args.batch, args.nhead_q, args.seqlen_q, args.hdim_v);
    constexpr dim3 blocks = Kernel::BlockSize();
    constexpr index_t kBlockPerCu = Kernel::kBlockPerCu;

    return launch_kernel(config, make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

// return value:
//   first  = whether the kernel was launched (true = launched, false = skipped)
//   second = elapsed time (ms) of the kernel launch, valid only if first == true
template <typename KernelTraits>
std::pair<bool, float> fmha_fwd_v3_kernel_dispatch(const fmha_fwd_v3_args& args,
                                                   const stream_config& config);

} // namespace ck_tile
