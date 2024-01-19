// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/tile_program/block_tile/block_masking.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qs_ks_vs.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"
#include "ck/tile_program/tile/tile_fmha_traits.hpp"

#include "fmha_fwd_epilogue.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<ck::half_t>
{
    using QDataType           = ck::half_t;
    using KDataType           = ck::half_t;
    using VDataType           = ck::half_t;
    using BiasDataType        = ck::half_t;
    using LSEDataType         = float;      // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float;      // data type for first gemm accumulation
    using SMPLComputeDataType = float;      // data type for reduction, softmax
    using PDataType           = ck::half_t; // data type for A matrix of second gemm
    using OaccDataType        = float;      // data type for second gemm accumulation
    using ODataType           = ck::half_t;
};

template <>
struct FmhaFwdTypeConfig<ck::bhalf_t>
{
    using QDataType           = ck::bhalf_t;
    using KDataType           = ck::bhalf_t;
    using VDataType           = ck::bhalf_t;
    using BiasDataType        = ck::bhalf_t;
    using LSEDataType         = float;       // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float;       // data type for first gemm accumulation
    using SMPLComputeDataType = float;       // data type for reduction, softmax
    using PDataType           = ck::bhalf_t; // data type for A matrix of second gemm
    using OaccDataType        = float;       // data type for second gemm accumulation
    using ODataType           = ck::bhalf_t;
};

// default settings for FmhaFwdKernelSelector<> type alias
using VLayout = ck::tensor_layout::gemm::RowMajor; // (bs, nhead) seqlen * hdim
// using VLayout = ck::tensor_layout::gemm::ColumnMajor; // (bs, nhead) hdim * seqlen

struct FmhaMasks
{
    using NoMask      = ck::tile_program::block::GenericAttentionMask<false>;
    using GenericMask = ck::tile_program::block::GenericAttentionMask<true, true>;
    using CausalMask  = ck::tile_program::block::GenericAttentionMask<true, false>;
};

inline constexpr bool kM0NeedPadding   = false;
inline constexpr bool kN0K1NeedPadding = false;
inline constexpr bool kK0N1NeedPadding = false;

template <ck::index_t HDim>
struct FmhaBlockTile;

template <>
struct FmhaBlockTile</* HDim = */ 32> : ck::Sequence<128, 64, 16, 32, 32, 32>
{
};
template <>
struct FmhaBlockTile</* HDim = */ 64> : ck::Sequence<128, 64, 32, 64, 32, 64>
{
};
template <>
struct FmhaBlockTile</* HDim = */ 128> : ck::Sequence<128, 128, 32, 128, 32, 128>
{
};
template <>
struct FmhaBlockTile</* HDim = */ 256> : ck::Sequence<128, 128, 32, 256, 32, 256>
{
};
using FmhaBlockWarps = ck::Sequence<4, 1, 1>;
using FmhaWarpTile   = ck::Sequence<32, 32, 16>;

template <ck::index_t HDim>
struct FmhaShape;

template <>
struct FmhaShape</* HDim = */ 32> : ck::tile_program::TileFmhaShape<FmhaBlockTile</* HDim = */ 32>,
                                                                    ck::Sequence<2, 1, 1>,
                                                                    FmhaWarpTile,
                                                                    ck::Sequence<2, 1, 1>,
                                                                    FmhaWarpTile,
                                                                    VLayout>
{
};

template <>
struct FmhaShape</* HDim = */ 64> : ck::tile_program::TileFmhaShape<FmhaBlockTile</* HDim = */ 64>,
                                                                    FmhaBlockWarps,
                                                                    FmhaWarpTile,
                                                                    FmhaBlockWarps,
                                                                    FmhaWarpTile,
                                                                    VLayout>
{
};

template <>
struct FmhaShape</* HDim = */ 128>
    : ck::tile_program::TileFmhaShape<FmhaBlockTile</* HDim = */ 128>,
                                      FmhaBlockWarps,
                                      FmhaWarpTile,
                                      FmhaBlockWarps,
                                      FmhaWarpTile,
                                      VLayout>
{
};

template <>
struct FmhaShape</* HDim = */ 256>
    : ck::tile_program::TileFmhaShape<FmhaBlockTile</* HDim = */ 256>,
                                      FmhaBlockWarps,
                                      FmhaWarpTile,
                                      FmhaBlockWarps,
                                      FmhaWarpTile,
                                      VLayout>
{
};

template <ck::index_t HDim, bool kHasBias, bool kStoreLSE>
using FmhaTraits = ck::tile_program::TileFmhaTraits<kM0NeedPadding,
                                                    kN0K1NeedPadding,
                                                    kK0N1NeedPadding,
                                                    kHasBias,
                                                    kStoreLSE,
                                                    HDim == 64 ? /* occupancy = */ 3 : 2>;

template <ck::index_t HDim,
          typename DataType,
          bool kIsGroupMode,
          typename FmhaMask,
          bool kHasBias,
          bool kStoreLSE>
using FmhaPipelineProblem = ck::tile_program::block::BlockFmhaPipelineProblem<
    typename FmhaFwdTypeConfig<DataType>::QDataType,
    typename FmhaFwdTypeConfig<DataType>::KDataType,
    typename FmhaFwdTypeConfig<DataType>::VDataType,
    typename FmhaFwdTypeConfig<DataType>::SaccDataType,
    typename FmhaFwdTypeConfig<DataType>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<DataType>::BiasDataType,
    typename FmhaFwdTypeConfig<DataType>::LSEDataType,
    typename FmhaFwdTypeConfig<DataType>::PDataType,
    typename FmhaFwdTypeConfig<DataType>::OaccDataType,
    typename FmhaFwdTypeConfig<DataType>::ODataType,
    /* BlockSize = */ HDim == 32 ? 128 : 256,
    FmhaShape<HDim>,
    kIsGroupMode,
    FmhaMask,
    FmhaTraits<HDim, kHasBias, kStoreLSE>>;

template <ck::index_t HDim,
          typename DataType,
          bool kIsGroupMode,
          typename FmhaMask,
          bool kHasBias,
          bool kStoreLSE>
using FmhaPipeline = ck::tile_program::block::BlockFmhaPipelineQRKSVSAsync<
    FmhaPipelineProblem<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias, kStoreLSE>>;

template <typename DataType>
using FmhaEpilogue =
    FmhaFwdEpilogue<FmhaFwdEpilogueProblem<typename FmhaFwdTypeConfig<DataType>::OaccDataType,
                                           typename FmhaFwdTypeConfig<DataType>::ODataType>>;

template <ck::index_t HDim,
          typename DataType,
          bool kIsGroupMode,
          typename FmhaMask,
          bool kHasBias,
          bool kStoreLSE>
using FmhaFwdKernelSelector =
    FmhaFwdKernel<FmhaFwdTilePartitioner<FmhaShape<HDim>>,
                  FmhaPipeline<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias, kStoreLSE>,
                  FmhaEpilogue<DataType>>;

// Kernel API
template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(const void* q_ptr,
                                     const void* k_ptr,
                                     const void* v_ptr,
                                     const void* bias_ptr,
                                     void* lse_ptr,
                                     void* o_ptr,
                                     const void* seqstart_q_ptr,
                                     const void* seqstart_k_ptr,
                                     const void* seqlen_k_ptr,
                                     ck::index_t batch,
                                     ck::index_t nhead,
                                     ck::index_t nhead_k,
                                     ck::index_t seqlen_q,
                                     ck::index_t seqlen_k,
                                     ck::index_t hdim_q,
                                     ck::index_t hdim_v,
                                     ck::index_t max_seqlen_q,
                                     float scale,
                                     bool i_perm,
                                     bool o_perm,
                                     ck::index_t mask_y,
                                     ck::index_t mask_x)
{
    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernel::VLayout, ck::tensor_layout::gemm::RowMajor>;

    assert(nhead % nhead_k == 0);
    /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' & 'nhead_stride_bias'
    ///       are 0.
    // setup stride_* arguments
    const ck::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
    const ck::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
    const ck::index_t stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? hdim_v : nhead_k * hdim_v;
        else
            return i_perm ? seqlen_k : nhead_k * seqlen_k;
    }();
    const ck::index_t stride_bias = (i_perm ? seqlen_k : 1 * seqlen_k);
    const ck::index_t stride_o    = (o_perm ? hdim_v : nhead * hdim_v);
    // setup nhead_stride_* arguments
    const ck::index_t nhead_stride_q = (i_perm ? seqlen_q * hdim_q : hdim_q);
    const ck::index_t nhead_stride_k = (i_perm ? seqlen_k * hdim_q : hdim_q);
    const ck::index_t nhead_stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? seqlen_k * hdim_v : hdim_v;
        else
            return i_perm ? hdim_v * seqlen_k : seqlen_k;
    }();
    const ck::index_t nhead_stride_bias = (i_perm ? 0 * seqlen_q * seqlen_k : 0 * seqlen_k);
    const ck::index_t nhead_stride_lse  = (seqlen_q * 1);
    const ck::index_t nhead_stride_o    = (o_perm ? seqlen_q * hdim_v : hdim_v);
    // setup batch_stride_* arguments
    const ck::index_t batch_stride_q    = (nhead * seqlen_q * hdim_q);
    const ck::index_t batch_stride_k    = (nhead_k * seqlen_k * hdim_q);
    const ck::index_t batch_stride_v    = (nhead_k * hdim_v * seqlen_k);
    const ck::index_t batch_stride_bias = (0 * nhead * seqlen_q * seqlen_k);
    const ck::index_t batch_stride_lse  = (nhead * seqlen_q * 1);
    const ck::index_t batch_stride_o    = (nhead * seqlen_q * hdim_v);

    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         bias_ptr,
                                         lse_ptr,
                                         o_ptr,
                                         seqstart_q_ptr,
                                         seqstart_k_ptr,
                                         seqlen_k_ptr,
                                         hdim_q,
                                         hdim_v,
                                         nhead / nhead_k,
                                         scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_bias,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_bias,
                                         nhead_stride_lse,
                                         nhead_stride_o,
                                         mask_y,
                                         mask_x);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargs(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         bias_ptr,
                                         lse_ptr,
                                         o_ptr,
                                         seqlen_q,
                                         seqlen_k,
                                         hdim_q,
                                         hdim_v,
                                         nhead / nhead_k,
                                         scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_bias,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_bias,
                                         nhead_stride_lse,
                                         nhead_stride_o,
                                         batch_stride_q,
                                         batch_stride_k,
                                         batch_stride_v,
                                         batch_stride_bias,
                                         batch_stride_lse,
                                         batch_stride_o,
                                         mask_y,
                                         mask_x);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(batch, nhead, max_seqlen_q, hdim_v);
    return ck::make_tuple(kargs, grids);
}

// will instantiate this function across different source file
template <typename FmhaKernel>
float fmha_fwd_run(const StreamConfig&, typename FmhaKernel::Kargs, dim3);

#define FMHA_FWD_KERNEL_DEFINE(KERNEL_)                                                          \
    template <>                                                                                  \
    float fmha_fwd_run<KERNEL_>(                                                                 \
        const StreamConfig& stream, typename KERNEL_::Kargs kargs, dim3 grids)                   \
    {                                                                                            \
        constexpr dim3 blocks             = KERNEL_::BlockSize();                                \
        constexpr ck::index_t kBlockPerCu = KERNEL_::kBlockPerCu;                                \
        return launch_kernel<blocks.x, kBlockPerCu>(stream, KERNEL_{}, grids, blocks, 0, kargs); \
    }
