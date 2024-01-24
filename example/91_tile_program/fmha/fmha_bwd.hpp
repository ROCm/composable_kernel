// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/tile_program/block_tile/block_masking.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_dispatcher.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_dot_do_o.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"
#include "ck/tile_program/tile/tile_fmha_traits.hpp"

#include "fmha_bwd_epilogue.hpp"
#include "fmha_bwd_kernel.hpp"
#include "fmha_bwd_tile_partitioner.hpp"

template <typename DataType>
struct FmhaBwdTypeConfig;

template <>
struct FmhaBwdTypeConfig<ck::half_t>
{
    using QDataType        = ck::half_t;
    using KDataType        = ck::half_t;
    using VDataType        = ck::half_t;
    using GemmDataType     = ck::half_t;
    using BiasDataType     = ck::half_t;
    using LSEDataType      = float;
    using AccDataType      = float; // data type for gemm accumulation
    using DDataType        = float;
    using ZDataType        = unsigned short;
    using ODataType        = ck::half_t;
    using OGradDataType    = ck::half_t;
    using QGradDataType    = ck::half_t;
    using KGradDataType    = ck::half_t;
    using VGradDataType    = ck::half_t;
    using BiasGradDataType = ck::half_t;
};

template <>
struct FmhaBwdTypeConfig<ck::bhalf_t>
{
    using QDataType        = ck::bhalf_t;
    using KDataType        = ck::bhalf_t;
    using VDataType        = ck::bhalf_t;
    using GemmDataType     = ck::bhalf_t;
    using BiasDataType     = ck::bhalf_t;
    using LSEDataType      = float;
    using AccDataType      = float; // data type for gemm accumulation
    using DDataType        = float;
    using ZDataType        = unsigned short;
    using ODataType        = ck::bhalf_t;
    using OGradDataType    = ck::bhalf_t;
    using QGradDataType    = ck::bhalf_t;
    using KGradDataType    = ck::bhalf_t;
    using VGradDataType    = ck::bhalf_t;
    using BiasGradDataType = ck::bhalf_t;
};

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

// GEMM0: Q@K=S^T
// GEMM1: P^T@dO^T=dV(This was chosen as G1 to match fwd, but N1 must be equal to headdim_v)
// GEMM2: dO@V=dP^T(This was chosen as G2 because of the calculation order)
// GEMM3: dS^T@Q^T=dK(Similar to G1, but N3 must be equal to headdim_qk)
// GEMM4: dS@K^T=dQ(N4 must be equal to headdim_qk)
// Is it necessary to distinguish between K0~K4?
// clang-format off
// #################################################|  M0|  N0|  K0|  K1|  K2|  K3|  K4| QKHD|  VHD|
template <>
struct FmhaBlockTile</* HDim = */ 32>  : ck::Sequence<128, 128,  32,  32,  32,  32,  32,   32,   32>
{
};
template <>
struct FmhaBlockTile</* HDim = */ 64>  : ck::Sequence< 64, 128,  32,  32,  32,  32,  32,   64,   64>
{
};
template <>
struct FmhaBlockTile</* HDim = */ 128> : ck::Sequence< 64, 128,  32,  32,  32,  32,  32,  128,  128>
{
};
// clang-format on

template <ck::index_t HDim>
struct FmhaLoadStrategy;

// clang-format off
// ######################################################| QLoadOnce| QTLoadOnce| KLoadOnce| KTLoadOnce| VLoadOnce| OGradLoadOnce| OGradTLoadOnce|
template <>
struct FmhaLoadStrategy</* HDim = */ 32>   : ck::Sequence<      true,      false,      true,      false,      true,          true,          false> // 10
// struct FmhaLoadStrategy</* HDim = */ 32>   : ck::Sequence<     false,      false,      true,       true,      true,         false,          false> // 9
{
};
template <>
struct FmhaLoadStrategy</* HDim = */ 64>   : ck::Sequence<     false,      false,      true,       true,      true,         false,          false> // 9
// struct FmhaLoadStrategy</* HDim = */ 64>   : ck::Sequence<      true,      false,      true,      false,      true,          true,          false> // 10
{
};
template <>
struct FmhaLoadStrategy</* HDim = */ 128>  : ck::Sequence<     false,      false,      true,      false,      true,         false,          false> // 13
{
};
// clang-format on

using FmhaBlockWarps0 = ck::Sequence<1, 4, 1>;
using FmhaBlockWarps1 = ck::Sequence<4, 1, 1>;
using FmhaBlockWarps2 = ck::Sequence<2, 2, 1>;
using FmhaWarpTile0   = ck::Sequence<32, 32, 16>;
using FmhaWarpTile1   = ck::Sequence<16, 16, 16>;
// TODO: simplify Gemm0~4BlockWarps in TileFmhaBwdShape
//       G0&G2 -> GSdP
//       G1&G3 -> GdKV
//       G4    -> GdQ
template <ck::index_t HDim>
struct FmhaBwdShape;

template <>
struct FmhaBwdShape</* HDim = */ 32>
    : ck::tile_program::TileFmhaBwdShape<FmhaBlockTile</* HDim = */ 32>,
                                         FmhaLoadStrategy</* HDim = */ 32>,
                                         FmhaBlockWarps0,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps1,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps0,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps1,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps1,
                                         FmhaWarpTile0>
{
};
template <>
struct FmhaBwdShape</* HDim = */ 64>
    : ck::tile_program::TileFmhaBwdShape<FmhaBlockTile</* HDim = */ 64>,
                                         FmhaLoadStrategy</* HDim = */ 64>,
                                         FmhaBlockWarps0,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps1,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps0,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps1,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps2,
                                         FmhaWarpTile0>
{
};
template <>
struct FmhaBwdShape</* HDim = */ 128>
    : ck::tile_program::TileFmhaBwdShape<FmhaBlockTile</* HDim = */ 128>,
                                         FmhaLoadStrategy</* HDim = */ 128>,
                                         FmhaBlockWarps0,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps1,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps0,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps1,
                                         FmhaWarpTile0,
                                         FmhaBlockWarps2,
                                         FmhaWarpTile0>
{
};

template <bool kHasBias>
using FmhaBwdTraits = ck::tile_program::
    TileFmhaTraits<kM0NeedPadding, kN0K1NeedPadding, kK0N1NeedPadding, kHasBias, 1>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode, typename FmhaMask, bool kHasBias>
using FmhaBwdPipelineProblem = ck::tile_program::block::BlockFmhaBwdPipelineProblem<
    typename FmhaBwdTypeConfig<DataType>::QDataType,
    typename FmhaBwdTypeConfig<DataType>::KDataType,
    typename FmhaBwdTypeConfig<DataType>::VDataType,
    typename FmhaBwdTypeConfig<DataType>::GemmDataType,
    typename FmhaBwdTypeConfig<DataType>::LSEDataType,
    typename FmhaBwdTypeConfig<DataType>::AccDataType,
    typename FmhaBwdTypeConfig<DataType>::DDataType,
    typename FmhaBwdTypeConfig<DataType>::ZDataType,
    typename FmhaBwdTypeConfig<DataType>::BiasDataType,
    typename FmhaBwdTypeConfig<DataType>::ODataType,
    typename FmhaBwdTypeConfig<DataType>::OGradDataType,
    typename FmhaBwdTypeConfig<DataType>::QGradDataType,
    typename FmhaBwdTypeConfig<DataType>::KGradDataType,
    typename FmhaBwdTypeConfig<DataType>::VGradDataType,
    typename FmhaBwdTypeConfig<DataType>::BiasGradDataType,
    /* BlockSize = */ 256,
    FmhaBwdShape<HDim>,
    kIsGroupMode,
    FmhaMask,
    FmhaBwdTraits<kHasBias>>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode, typename FmhaMask, bool kHasBias>
using FmhaBwdPipeline = typename ck::tile_program::block::BlockFmhaBwdPipelineDispatcher<
    FmhaLoadStrategy<HDim>,
    FmhaBwdPipelineProblem<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias>>::BlockPipeline;

template <typename DataType>
using FmhaEpilogue =
    FmhaBwdEpilogue<FmhaBwdEpilogueProblem<typename FmhaBwdTypeConfig<DataType>::AccDataType,
                                           typename FmhaBwdTypeConfig<DataType>::KGradDataType,
                                           typename FmhaBwdTypeConfig<DataType>::VGradDataType>>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode, typename FmhaMask, bool kHasBias>
using FmhaBwdKernelSelector =
    FmhaBwdKernel<FmhaBwdTilePartitioner<FmhaBwdShape<HDim>>,
                  FmhaBwdPipeline<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias>,
                  FmhaEpilogue<DataType>>;

using FmhaBwdOGradDotOTraits =
    ck::tile_program::TileFmhaBwdOGradDotOTraits<kM0NeedPadding, kK0N1NeedPadding>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode>
using FmhaBwdOGradDotOPipelineProblem =
    ck::tile_program::block::BlockFmhaBwdOGradDotOPipelineProblem<
        typename FmhaBwdTypeConfig<DataType>::ODataType,
        typename FmhaBwdTypeConfig<DataType>::OGradDataType,
        typename FmhaBwdTypeConfig<DataType>::DDataType,
        /* BlockSize = */ 256,
        FmhaBwdShape<HDim>::kVHeaddim,
        kIsGroupMode,
        FmhaBwdOGradDotOTraits>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode>
using FmhaBwdOGradDotO = ck::tile_program::block::BlockFmhaBwdOGradDotO<
    FmhaBwdOGradDotOPipelineProblem<HDim, DataType, kIsGroupMode>>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode>
using FmhaBwdOGradDotOKernelSelector =
    FmhaBwdOGradDotOKernel<FmhaBwdOGradDotOTilePartitioner</* BlockSize = */ 256>,
                           FmhaBwdOGradDotO<HDim, DataType, kIsGroupMode>>;

// Kernel API
template <typename FmhaBwdKernel>
auto fmha_bwd_create_kargs_and_grids(const void* q_ptr,
                                     const void* k_ptr,
                                     const void* v_ptr,
                                     const void* bias_ptr,
                                     const void* lse_ptr,
                                     const void* do_ptr,
                                     const void* d_ptr,
                                     // void* z_ptr,
                                     void* dq_ptr,
                                     void* dk_ptr,
                                     void* dv_ptr,
                                     void* dbias_ptr,
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
                                     ck::index_t max_seqlen_k,
                                     float scale,
                                     bool i_perm,
                                     bool o_perm,
                                     ck::index_t mask_y,
                                     ck::index_t mask_x)
{
    assert(nhead % nhead_k == 0);
    /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' & 'nhead_stride_bias'
    ///       are 0.
    // setup stride_* arguments
    const ck::index_t stride_q     = (i_perm ? hdim_q : nhead * hdim_q);
    const ck::index_t stride_k     = (i_perm ? hdim_q : nhead_k * hdim_q);
    const ck::index_t stride_v     = (i_perm ? hdim_v : nhead_k * hdim_v);
    const ck::index_t stride_bias  = (i_perm ? seqlen_k : 1 * seqlen_k);
    const ck::index_t stride_do    = (o_perm ? hdim_v : nhead * hdim_v);
    const ck::index_t stride_dk    = (i_perm ? hdim_q : nhead * hdim_q);
    const ck::index_t stride_dv    = (i_perm ? hdim_v : nhead * hdim_v);
    const ck::index_t stride_dbias = (i_perm ? seqlen_k : nhead * seqlen_k);
    // setup nhead_stride_* arguments
    const ck::index_t nhead_stride_q     = (i_perm ? seqlen_q * hdim_q : hdim_q);
    const ck::index_t nhead_stride_k     = (i_perm ? seqlen_k * hdim_q : hdim_q);
    const ck::index_t nhead_stride_v     = (i_perm ? seqlen_k * hdim_v : hdim_v);
    const ck::index_t nhead_stride_bias  = (i_perm ? 0 * seqlen_q * seqlen_k : 0 * seqlen_k);
    const ck::index_t nhead_stride_do    = (o_perm ? seqlen_q * hdim_v : hdim_v);
    const ck::index_t nhead_stride_lsed  = (seqlen_q);
    const ck::index_t nhead_stride_dbias = (i_perm ? seqlen_q * seqlen_k : seqlen_k);
    // setup batch_stride_* arguments
    const ck::index_t batch_stride_q     = (nhead * seqlen_q * hdim_q);
    const ck::index_t batch_stride_k     = (nhead_k * seqlen_k * hdim_q);
    const ck::index_t batch_stride_v     = (nhead_k * seqlen_k * hdim_v);
    const ck::index_t batch_stride_bias  = (0 * nhead * seqlen_q * seqlen_k);
    const ck::index_t batch_stride_do    = (nhead * seqlen_q * hdim_v);
    const ck::index_t batch_stride_lsed  = (nhead * seqlen_q);
    const ck::index_t batch_stride_dk    = (nhead * seqlen_k * hdim_q);
    const ck::index_t batch_stride_dv    = (nhead * seqlen_k * hdim_v);
    const ck::index_t batch_stride_dbias = (nhead * seqlen_q * seqlen_k);

    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdKernel::kIsGroupMode)
        {
            return FmhaBwdKernel::MakeKargs(q_ptr,
                                            k_ptr,
                                            v_ptr,
                                            bias_ptr,
                                            lse_ptr,
                                            do_ptr,
                                            d_ptr,
                                            dq_ptr,
                                            dk_ptr,
                                            dv_ptr,
                                            dbias_ptr,
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
                                            stride_do,
                                            stride_dk,
                                            stride_dv,
                                            stride_dbias,
                                            nhead_stride_q,
                                            nhead_stride_k,
                                            nhead_stride_v,
                                            nhead_stride_bias,
                                            nhead_stride_do,
                                            nhead_stride_lsed,
                                            nhead_stride_dbias,
                                            mask_y,
                                            mask_x);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdKernel::MakeKargs(q_ptr,
                                            k_ptr,
                                            v_ptr,
                                            bias_ptr,
                                            lse_ptr,
                                            do_ptr,
                                            d_ptr,
                                            dq_ptr,
                                            dk_ptr,
                                            dv_ptr,
                                            dbias_ptr,
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
                                            stride_do,
                                            stride_dk,
                                            stride_dv,
                                            stride_dbias,
                                            nhead_stride_q,
                                            nhead_stride_k,
                                            nhead_stride_v,
                                            nhead_stride_bias,
                                            nhead_stride_do,
                                            nhead_stride_lsed,
                                            nhead_stride_dbias,
                                            batch_stride_q,
                                            batch_stride_k,
                                            batch_stride_v,
                                            batch_stride_bias,
                                            batch_stride_do,
                                            batch_stride_lsed,
                                            batch_stride_dk,
                                            batch_stride_dv,
                                            batch_stride_dbias,
                                            mask_y,
                                            mask_x);
        }
    }();

    dim3 grids = FmhaBwdKernel::GridSize(batch, nhead, max_seqlen_k);
    return ck::make_tuple(kargs, grids);
}

template <typename FmhaBwdOGradDotOKernel>
auto fmha_bwd_dot_do_o_create_kargs_and_grids(const void* o_ptr,
                                              const void* do_ptr,
                                              void* d_ptr,
                                              const void* seqstart_q_ptr,
                                              ck::index_t batch,
                                              ck::index_t nhead,
                                              ck::index_t seqlen_q,
                                              ck::index_t hdim_v,
                                              ck::index_t max_seqlen_q,
                                              bool o_perm)
{
    const ck::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
    const ck::index_t nhead_stride_o = (o_perm ? seqlen_q * hdim_v : hdim_v);
    const ck::index_t nhead_stride_d = (seqlen_q);
    const ck::index_t batch_stride_o = (nhead * seqlen_q * hdim_v);
    const ck::index_t batch_stride_d = (nhead * seqlen_q);

    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdOGradDotOKernel::kIsGroupMode)
        {
            return FmhaBwdOGradDotOKernel::MakeKargs(o_ptr,
                                                     do_ptr,
                                                     d_ptr,
                                                     seqstart_q_ptr,
                                                     hdim_v,
                                                     stride_o,
                                                     nhead_stride_o,
                                                     nhead_stride_d);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdOGradDotOKernel::MakeKargs(o_ptr,
                                                     do_ptr,
                                                     d_ptr,
                                                     seqlen_q,
                                                     hdim_v,
                                                     stride_o,
                                                     nhead_stride_o,
                                                     nhead_stride_d,
                                                     batch_stride_o,
                                                     batch_stride_d);
        }
    }();

    dim3 grids = FmhaBwdOGradDotOKernel::GridSize(batch, nhead, max_seqlen_q);
    return ck::make_tuple(kargs, grids);
}

// will instantiate this function across different source file
template <typename FmhaBwdKernel>
float fmha_bwd_run(const StreamConfig&, typename FmhaBwdKernel::Kargs, dim3);

#define FMHA_BWD_KERNEL_DEFINE(KERNEL_)                                                          \
    template <>                                                                                  \
    float fmha_bwd_run<KERNEL_>(                                                                 \
        const StreamConfig& stream, typename KERNEL_::Kargs kargs, dim3 grids)                   \
    {                                                                                            \
        constexpr dim3 blocks             = KERNEL_::BlockSize();                                \
        constexpr ck::index_t kBlockPerCu = KERNEL_::kBlockPerCu;                                \
        return launch_kernel<blocks.x, kBlockPerCu>(stream, KERNEL_{}, grids, blocks, 0, kargs); \
    }

template <typename FmhaBwdOGradDotOKernel>
float fmha_bwd_dot_do_o_run(const StreamConfig&, typename FmhaBwdOGradDotOKernel::Kargs, dim3);

#define FMHA_BWD_DOT_DO_O_KERNEL_DEFINE(KERNEL_)                                                 \
    template <>                                                                                  \
    float fmha_bwd_dot_do_o_run<KERNEL_>(                                                        \
        const StreamConfig& stream, typename KERNEL_::Kargs kargs, dim3 grids)                   \
    {                                                                                            \
        constexpr dim3 blocks             = KERNEL_::BlockSize();                                \
        constexpr ck::index_t kBlockPerCu = KERNEL_::kBlockPerCu;                                \
        return launch_kernel<blocks.x, kBlockPerCu>(stream, KERNEL_{}, grids, blocks, 0, kargs); \
    }
