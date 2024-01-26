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
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qs_ks_vs.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"
#include "ck/tile_program/tile/tile_fmha_traits.hpp"

#include "fmha_fwd_epilogue.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"
#include "mask.hpp"

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

template <>
struct FmhaFwdTypeConfig<ck::f8_t>
{
    using QDataType           = ck::f8_t;
    using KDataType           = ck::f8_t;
    using VDataType           = ck::f8_t;
    using BiasDataType        = float;    // TODO: fix me
    using LSEDataType         = float;    // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float;    // data type for first gemm accumulation
    using SMPLComputeDataType = float;    // data type for reduction, softmax
    using PDataType           = ck::f8_t; // data type for A matrix of second gemm
    using OaccDataType        = float;    // data type for second gemm accumulation
    using ODataType           = ck::f8_t;
};

template <>
struct FmhaFwdTypeConfig<ck::bf8_t>
{
    using QDataType           = ck::bf8_t;
    using KDataType           = ck::bf8_t;
    using VDataType           = ck::bf8_t;
    using BiasDataType        = ck::bf8_t;
    using LSEDataType         = float;     // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float;     // data type for first gemm accumulation
    using SMPLComputeDataType = float;     // data type for reduction, softmax
    using PDataType           = ck::bf8_t; // data type for A matrix of second gemm
    using OaccDataType        = float;     // data type for second gemm accumulation
    using ODataType           = ck::bf8_t;
};

struct FmhaMasks
{
    using NoMask      = ck::tile_program::block::GenericAttentionMask<false>;
    using GenericMask = ck::tile_program::block::GenericAttentionMask<true, true>;
    using CausalMask  = ck::tile_program::block::GenericAttentionMask<true, false>;
};

// internal API, don't use this directly
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
                                     float descale_qk,
                                     float descale_sv,
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
                                         mask_x,
                                         descale_qk,
                                         descale_sv);
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
                                         mask_x,
                                         descale_qk,
                                         descale_sv);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(batch, nhead, max_seqlen_q, hdim_v);
    return ck::make_tuple(kargs, grids);
}

// This is the args from caller to underneath API, different from the kernel
struct fmha_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    void* lse_ptr;
    void* o_ptr;
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    ck::index_t batch;
    ck::index_t nhead;
    ck::index_t nhead_k;
    ck::index_t seqlen_q;
    ck::index_t seqlen_k;
    ck::index_t hdim_q;
    ck::index_t hdim_v;
    ck::index_t max_seqlen_q;
    float scale;
    float descale_qk;
    float descale_sv;
    bool i_perm;
    bool o_perm;
    ck::index_t mask_y;
    ck::index_t mask_x;
};

template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_fwd_args args)
{
    return fmha_fwd_create_kargs_and_grids<FmhaKernel>(args.q_ptr,
                                                       args.k_ptr,
                                                       args.v_ptr,
                                                       args.bias_ptr,
                                                       args.lse_ptr,
                                                       args.o_ptr,
                                                       args.seqstart_q_ptr,
                                                       args.seqstart_k_ptr,
                                                       args.seqlen_k_ptr,
                                                       args.batch,
                                                       args.nhead,
                                                       args.nhead_k,
                                                       args.seqlen_q,
                                                       args.seqlen_k,
                                                       args.hdim_q,
                                                       args.hdim_v,
                                                       args.max_seqlen_q,
                                                       args.scale,
                                                       args.descale_qk,
                                                       args.descale_sv,
                                                       args.i_perm,
                                                       args.o_perm,
                                                       args.mask_y,
                                                       args.mask_x);
}

// this is internal API, will be generated across different files to speedup compile
template <ck::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          bool kIsVLayoutRowMajor_,
          typename FmhaMask_,
          bool kHasBias_,
          bool kStoreLse_,
          bool kUseDropout_>
struct fmha_fwd_traits_
{
    static constexpr ck::index_t HDim        = HDim_;
    using DataType                           = ck::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode       = kIsGroupMode_;
    static constexpr bool kIsVLayoutRowMajor = kIsVLayoutRowMajor_;
    using FmhaMask                           = ck::remove_cvref_t<FmhaMask_>;
    static constexpr bool kHasBias           = kHasBias_;
    static constexpr bool kStoreLse          = kStoreLse_;
    static constexpr bool kUseDropout        = kUseDropout_;
};

template <typename Traits_>
float fmha_fwd_(const StreamConfig&, fmha_fwd_args);

// This is the public API, will be generated by script
struct fmha_fwd_traits
{
    int hdim;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    mask_enum mask_type;
    bool has_bias;
    bool has_lse;
    bool has_drop;
};
float fmha_fwd(fmha_fwd_traits, fmha_fwd_args, const StreamConfig&);
