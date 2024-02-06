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
#include "mask.hpp"

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

// internal API, don't use this directly
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

// This is the args from caller to underneath API, different from the kernel
struct fmha_bwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    const void* lse_ptr;
    const void* do_ptr;
    const void* d_ptr;
    // void* z_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* dbias_ptr;
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
    ck::index_t max_seqlen_k;
    float scale;
    bool i_perm;
    bool o_perm;
    ck::index_t mask_y;
    ck::index_t mask_x;
};

template <typename FmhaBwdKernel>
auto fmha_bwd_create_kargs_and_grids(fmha_bwd_args args)
{
    return fmha_bwd_create_kargs_and_grids<FmhaBwdKernel>(args.q_ptr,
                                                          args.k_ptr,
                                                          args.v_ptr,
                                                          args.bias_ptr,
                                                          args.lse_ptr,
                                                          args.do_ptr,
                                                          args.d_ptr,
                                                          args.dq_ptr,
                                                          args.dk_ptr,
                                                          args.dv_ptr,
                                                          args.dbias_ptr,
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
                                                          args.max_seqlen_k,
                                                          args.scale,
                                                          args.i_perm,
                                                          args.o_perm,
                                                          args.mask_y,
                                                          args.mask_x);
}

struct fmha_bwd_dot_do_o_args
{
    const void* o_ptr;
    const void* do_ptr;
    void* d_ptr;
    const void* seqstart_q_ptr;
    ck::index_t batch;
    ck::index_t nhead;
    ck::index_t seqlen_q;
    ck::index_t hdim_v;
    ck::index_t max_seqlen_q;
    bool o_perm;
};

template <typename FmhaBwdOGradDotOKernel>
auto fmha_bwd_dot_do_o_create_kargs_and_grids(fmha_bwd_dot_do_o_args args)
{
    return fmha_bwd_dot_do_o_create_kargs_and_grids<FmhaBwdOGradDotOKernel>(args.o_ptr,
                                                                            args.do_ptr,
                                                                            args.d_ptr,
                                                                            args.seqstart_q_ptr,
                                                                            args.batch,
                                                                            args.nhead,
                                                                            args.seqlen_q,
                                                                            args.hdim_v,
                                                                            args.max_seqlen_q,
                                                                            args.o_perm);
}

// this is internal API, will be generated across different files to speedup compile
template <ck::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          bool kHasBias_>
struct fmha_bwd_traits_
{
    static constexpr ck::index_t HDim  = HDim_;
    using DataType                     = ck::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode = kIsGroupMode_;
    using FmhaMask                     = ck::remove_cvref_t<FmhaMask_>;
    static constexpr bool kHasBias     = kHasBias_;
};

template <typename Traits_>
float fmha_bwd_(const StreamConfig&, fmha_bwd_args);

template <ck::index_t HDim_, typename DataType_, bool kIsGroupMode_>
struct fmha_bwd_dot_do_o_traits_
{
    static constexpr ck::index_t HDim  = HDim_;
    using DataType                     = ck::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode = kIsGroupMode_;
};

template <typename Traits_>
float fmha_bwd_dot_do_o_(const StreamConfig&, fmha_bwd_dot_do_o_args);

// This is the public API, will be generated by script
struct fmha_bwd_traits
{
    int hdim;
    std::string data_type;
    bool is_group_mode;
    mask_enum mask_type;
    bool has_bias;
};
float fmha_bwd(fmha_bwd_traits, fmha_bwd_args, const StreamConfig&);

struct fmha_bwd_dot_do_o_traits
{
    int hdim;
    std::string data_type;
    bool is_group_mode;
};
float fmha_bwd_dot_do_o(fmha_bwd_dot_do_o_traits, fmha_bwd_dot_do_o_args, const StreamConfig&);
