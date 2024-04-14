// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/tile_program/block_tile/block_masking.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_problem.hpp"
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
    using QDataType             = ck::half_t;
    using KDataType             = ck::half_t;
    using VDataType             = ck::half_t;
    using GemmDataType          = ck::half_t;
    using BiasDataType          = ck::half_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck::half_t;
    using OGradDataType         = ck::half_t;
    using QGradDataType         = ck::half_t;
    using KGradDataType         = ck::half_t;
    using VGradDataType         = ck::half_t;
    using BiasGradDataType      = ck::half_t;
};

template <>
struct FmhaBwdTypeConfig<ck::bhalf_t>
{
    using QDataType             = ck::bhalf_t;
    using KDataType             = ck::bhalf_t;
    using VDataType             = ck::bhalf_t;
    using GemmDataType          = ck::bhalf_t;
    using BiasDataType          = ck::bhalf_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck::bhalf_t;
    using OGradDataType         = ck::bhalf_t;
    using QGradDataType         = ck::bhalf_t;
    using KGradDataType         = ck::bhalf_t;
    using VGradDataType         = ck::bhalf_t;
    using BiasGradDataType      = ck::bhalf_t;
};

struct FmhaMasks
{
    using NoMask      = ck::tile_program::block::GenericAttentionMask<false>;
    using GenericMask = ck::tile_program::block::GenericAttentionMask<true, true>;
    using CausalMask  = ck::tile_program::block::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct fmha_bwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    const void* lse_ptr;
    const void* do_ptr;
    const void* d_ptr;
    void* rand_val_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* dbias_ptr;
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    ck::index_t seqlen_q;
    ck::index_t seqlen_k;
    ck::index_t batch;
    ck::index_t max_seqlen_k;
    ck::index_t hdim_q;
    ck::index_t hdim_v;
    ck::index_t nhead_q;
    ck::index_t nhead_k;
    float scale;
    ck::index_t stride_q;
    ck::index_t stride_k;
    ck::index_t stride_v;
    ck::index_t stride_bias;
    ck::index_t stride_randval;
    ck::index_t stride_do;
    ck::index_t stride_dk;
    ck::index_t stride_dv;
    ck::index_t stride_dbias;
    ck::index_t nhead_stride_q;
    ck::index_t nhead_stride_k;
    ck::index_t nhead_stride_v;
    ck::index_t nhead_stride_bias;
    ck::index_t nhead_stride_randval;
    ck::index_t nhead_stride_do;
    ck::index_t nhead_stride_lsed;
    ck::index_t nhead_stride_dbias;
    ck::index_t batch_stride_q;
    ck::index_t batch_stride_k;
    ck::index_t batch_stride_v;
    ck::index_t batch_stride_bias;
    ck::index_t batch_stride_randval;
    ck::index_t batch_stride_do;
    ck::index_t batch_stride_lsed;
    ck::index_t batch_stride_dk;
    ck::index_t batch_stride_dv;
    ck::index_t batch_stride_dbias;
    ck::index_t window_size_left;
    ck::index_t window_size_right;
    ck::index_t mask_type;
    float p_drop;
    bool s_randval;
    std::tuple<uint64_t, uint64_t> drop_seed_offset;
};

template <typename FmhaBwdKernel>
auto fmha_bwd_create_kargs_and_grids(fmha_bwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdKernel::kIsGroupMode)
        {
            return FmhaBwdKernel::MakeKargs(args.q_ptr,
                                            args.k_ptr,
                                            args.v_ptr,
                                            args.bias_ptr,
                                            args.lse_ptr,
                                            args.do_ptr,
                                            args.d_ptr,
                                            args.rand_val_ptr,
                                            args.dq_ptr,
                                            args.dk_ptr,
                                            args.dv_ptr,
                                            args.dbias_ptr,
                                            args.seqstart_q_ptr,
                                            args.seqstart_k_ptr,
                                            args.seqlen_k_ptr,
                                            args.hdim_q,
                                            args.hdim_v,
                                            args.nhead_q,
                                            args.nhead_q / args.nhead_k,
                                            args.scale,
                                            args.stride_q,
                                            args.stride_k,
                                            args.stride_v,
                                            args.stride_bias,
                                            args.stride_randval,
                                            args.stride_do,
                                            args.stride_dk,
                                            args.stride_dv,
                                            args.stride_dbias,
                                            args.nhead_stride_q,
                                            args.nhead_stride_k,
                                            args.nhead_stride_v,
                                            args.nhead_stride_bias,
                                            args.nhead_stride_randval,
                                            args.nhead_stride_do,
                                            args.nhead_stride_lsed,
                                            args.nhead_stride_dbias,
                                            args.window_size_left,
                                            args.window_size_right,
                                            args.mask_type,
                                            args.p_drop,
                                            args.s_randval,
                                            args.drop_seed_offset);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdKernel::MakeKargs(args.q_ptr,
                                            args.k_ptr,
                                            args.v_ptr,
                                            args.bias_ptr,
                                            args.lse_ptr,
                                            args.do_ptr,
                                            args.d_ptr,
                                            args.rand_val_ptr,
                                            args.dq_ptr,
                                            args.dk_ptr,
                                            args.dv_ptr,
                                            args.dbias_ptr,
                                            args.seqlen_q,
                                            args.seqlen_k,
                                            args.hdim_q,
                                            args.hdim_v,
                                            args.nhead_q,
                                            args.nhead_q / args.nhead_k,
                                            args.scale,
                                            args.stride_q,
                                            args.stride_k,
                                            args.stride_v,
                                            args.stride_bias,
                                            args.stride_randval,
                                            args.stride_do,
                                            args.stride_dk,
                                            args.stride_dv,
                                            args.stride_dbias,
                                            args.nhead_stride_q,
                                            args.nhead_stride_k,
                                            args.nhead_stride_v,
                                            args.nhead_stride_bias,
                                            args.nhead_stride_randval,
                                            args.nhead_stride_do,
                                            args.nhead_stride_lsed,
                                            args.nhead_stride_dbias,
                                            args.batch_stride_q,
                                            args.batch_stride_k,
                                            args.batch_stride_v,
                                            args.batch_stride_bias,
                                            args.batch_stride_randval,
                                            args.batch_stride_do,
                                            args.batch_stride_lsed,
                                            args.batch_stride_dk,
                                            args.batch_stride_dv,
                                            args.batch_stride_dbias,
                                            args.window_size_left,
                                            args.window_size_right,
                                            args.mask_type,
                                            args.p_drop,
                                            args.s_randval,
                                            args.drop_seed_offset);
        }
    }();

    dim3 grids = FmhaBwdKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_k);
    return ck::make_tuple(kargs, grids);
}

struct fmha_bwd_dot_do_o_args
{
    const void* o_ptr;
    const void* do_ptr;
    void* d_ptr;
    const void* seqstart_q_ptr;
    float p_undrop;
    ck::index_t batch;
    ck::index_t nhead_q;
    ck::index_t seqlen_q;
    ck::index_t hdim_v;
    ck::index_t max_seqlen_q;
    ck::index_t stride_o;
    ck::index_t nhead_stride_o;
    ck::index_t nhead_stride_d;
    ck::index_t batch_stride_o;
    ck::index_t batch_stride_d;
};

template <typename FmhaBwdOGradDotOKernel>
auto fmha_bwd_dot_do_o_create_kargs_and_grids(fmha_bwd_dot_do_o_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdOGradDotOKernel::kIsGroupMode)
        {
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.p_undrop,
                                                     args.seqstart_q_ptr,
                                                     args.hdim_v,
                                                     args.stride_o,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_d);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.p_undrop,
                                                     args.seqlen_q,
                                                     args.hdim_v,
                                                     args.stride_o,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_d,
                                                     args.batch_stride_o,
                                                     args.batch_stride_d);
        }
    }();

    dim3 grids = FmhaBwdOGradDotOKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          bool kHasBias_,
          bool kHasBiasGrad_,
          bool kHasDropout_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_>
struct fmha_bwd_traits_
{
    static constexpr ck::index_t HDim  = HDim_;
    using DataType                     = ck::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode = kIsGroupMode_;
    using FmhaMask                     = ck::remove_cvref_t<FmhaMask_>;
    static constexpr bool kHasBias     = kHasBias_;
    static constexpr bool kHasBiasGrad = kHasBiasGrad_;
    static constexpr bool kHasDropout  = kHasDropout_;
    static constexpr bool kPadS        = kPadS_;
    static constexpr bool kPadSK       = kPadSK_;
    static constexpr bool kPadD        = kPadD_;
    static constexpr bool kPadDv       = kPadDv_;
};

template <typename Traits_>
float fmha_bwd_(const StreamConfig&, fmha_bwd_args);

template <ck::index_t HDim_, typename DataType_, bool kIsGroupMode_, bool kPadS_, bool kPadDv_>
struct fmha_bwd_dot_do_o_traits_
{
    static constexpr ck::index_t HDim  = HDim_;
    using DataType                     = ck::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode = kIsGroupMode_;
    static constexpr bool kPadS        = kPadS_;
    static constexpr bool kPadDv       = kPadDv_;
};

template <typename Traits_>
float fmha_bwd_dot_do_o_(const StreamConfig&, fmha_bwd_dot_do_o_args);

// This is the public API, will be generated by script
struct fmha_bwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    mask_enum mask_type;
    bool has_bias;
    bool has_dbias;
    bool has_dropout;
    // TODO: padding check is inside this api
};
float fmha_bwd(fmha_bwd_traits, fmha_bwd_args, const StreamConfig&);

struct fmha_bwd_dot_do_o_traits
{
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    // TODO: padding check is inside this api
};
float fmha_bwd_dot_do_o(fmha_bwd_dot_do_o_traits, fmha_bwd_dot_do_o_args, const StreamConfig&);
