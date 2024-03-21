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
    using QDataType             = ck::half_t;
    using KDataType             = ck::half_t;
    using VDataType             = ck::half_t;
    using BiasDataType          = ck::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;      // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float;      // data type for first gemm accumulation
    using SMPLComputeDataType   = float;      // data type for reduction, softmax
    using PDataType             = ck::half_t; // data type for A matrix of second gemm
    using OaccDataType          = float;      // data type for second gemm accumulation
    using ODataType             = ck::half_t;
};

template <>
struct FmhaFwdTypeConfig<ck::bhalf_t>
{
    using QDataType             = ck::bhalf_t;
    using KDataType             = ck::bhalf_t;
    using VDataType             = ck::bhalf_t;
    using BiasDataType          = ck::bhalf_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck::bhalf_t; // data type for A matrix of second gemm
    using OaccDataType          = float;       // data type for second gemm accumulation
    using ODataType             = ck::bhalf_t;
};

template <>
struct FmhaFwdTypeConfig<ck::f8_t>
{
    using QDataType             = ck::f8_t;
    using KDataType             = ck::f8_t;
    using VDataType             = ck::f8_t;
    using BiasDataType          = float; // TODO: fix me
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;    // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float;    // data type for first gemm accumulation
    using SMPLComputeDataType   = float;    // data type for reduction, softmax
    using PDataType             = ck::f8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;    // data type for second gemm accumulation
    using ODataType             = ck::f8_t;
};

template <>
struct FmhaFwdTypeConfig<ck::bf8_t>
{
    using QDataType             = ck::bf8_t;
    using KDataType             = ck::bf8_t;
    using VDataType             = ck::bf8_t;
    using BiasDataType          = ck::bf8_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;     // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float;     // data type for first gemm accumulation
    using SMPLComputeDataType   = float;     // data type for reduction, softmax
    using PDataType             = ck::bf8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;     // data type for second gemm accumulation
    using ODataType             = ck::bf8_t;
};

struct FmhaMasks
{
    using NoMask      = ck::tile_program::block::GenericAttentionMask<false>;
    using GenericMask = ck::tile_program::block::GenericAttentionMask<true, true>;
    using CausalMask  = ck::tile_program::block::GenericAttentionMask<true, false>;
};

#if 0
// internal API, don't use this directly
template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(const void* q_ptr,
                                     const void* k_ptr,
                                     const void* v_ptr,
                                     const void* bias_ptr,
                                     void* rand_val_ptr,
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
                                     ck::index_t max_seqlen_k,
                                     float scale,
                                     float descale_qk,
                                     float descale_sv,
                                     bool i_perm,
                                     bool o_perm,
                                     ck::index_t mask_y,
                                     ck::index_t mask_x,
                                     float p_drop,
                                     bool s_randval,
                                     std::tuple<uint64_t, uint64_t>& drop_seed_offset)
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
    const ck::index_t stride_bias    = (i_perm ? seqlen_k : 1 * seqlen_k);
    const ck::index_t stride_randval = (max_seqlen_k);
    const ck::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
    // setup nhead_stride_* arguments
    const ck::index_t nhead_stride_q = (i_perm ? seqlen_q * hdim_q : hdim_q);
    const ck::index_t nhead_stride_k = (i_perm ? seqlen_k * hdim_q : hdim_q);
    const ck::index_t nhead_stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? seqlen_k * hdim_v : hdim_v;
        else
            return i_perm ? hdim_v * seqlen_k : seqlen_k;
    }();
    const ck::index_t nhead_stride_bias    = (i_perm ? 0 * seqlen_q * seqlen_k : 0 * seqlen_k);
    const ck::index_t nhead_stride_randval = (seqlen_q * max_seqlen_k);
    const ck::index_t nhead_stride_lse     = (seqlen_q * 1);
    const ck::index_t nhead_stride_o       = (o_perm ? seqlen_q * hdim_v : hdim_v);
    // setup batch_stride_* arguments
    const ck::index_t batch_stride_q       = (nhead * seqlen_q * hdim_q);
    const ck::index_t batch_stride_k       = (nhead_k * seqlen_k * hdim_q);
    const ck::index_t batch_stride_v       = (nhead_k * hdim_v * seqlen_k);
    const ck::index_t batch_stride_bias    = (0 * nhead * seqlen_q * seqlen_k);
    const ck::index_t batch_stride_randval = (nhead * seqlen_q * max_seqlen_k);
    const ck::index_t batch_stride_lse     = (nhead * seqlen_q * 1);
    const ck::index_t batch_stride_o       = (nhead * seqlen_q * hdim_v);

    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         bias_ptr,
                                         rand_val_ptr,
                                         lse_ptr,
                                         o_ptr,
                                         seqstart_q_ptr,
                                         seqstart_k_ptr,
                                         seqlen_k_ptr,
                                         hdim_q,
                                         hdim_v,
                                         nhead,
                                         nhead / nhead_k,
                                         scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_bias,
                                         stride_randval,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_bias,
                                         nhead_stride_randval,
                                         nhead_stride_lse,
                                         nhead_stride_o,
                                         mask_y,
                                         mask_x,
                                         descale_qk,
                                         descale_sv,
                                         p_drop,
                                         s_randval,
                                         drop_seed_offset);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargs(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         bias_ptr,
                                         rand_val_ptr,
                                         lse_ptr,
                                         o_ptr,
                                         seqlen_q,
                                         seqlen_k,
                                         hdim_q,
                                         hdim_v,
                                         nhead,
                                         nhead / nhead_k,
                                         scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_bias,
                                         stride_randval,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_bias,
                                         nhead_stride_randval,
                                         nhead_stride_lse,
                                         nhead_stride_o,
                                         batch_stride_q,
                                         batch_stride_k,
                                         batch_stride_v,
                                         batch_stride_bias,
                                         batch_stride_randval,
                                         batch_stride_lse,
                                         batch_stride_o,
                                         mask_y,
                                         mask_x,
                                         descale_qk,
                                         descale_sv,
                                         p_drop,
                                         s_randval,
                                         drop_seed_offset);
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
    void* rand_val_ptr;
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
    ck::index_t max_seqlen_k;
    float scale;
    float descale_qk;
    float descale_sv;
    bool i_perm;
    bool o_perm;
    ck::index_t mask_y;
    ck::index_t mask_x;
    float p_drop;
    bool s_randval;
    std::tuple<uint64_t, uint64_t> drop_seed_offset;
};
#endif

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct fmha_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    ck::index_t seqlen_q;
    ck::index_t seqlen_k;
    ck::index_t batch;
    ck::index_t max_seqlen_q;
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
    ck::index_t stride_o;
    ck::index_t nhead_stride_q;
    ck::index_t nhead_stride_k;
    ck::index_t nhead_stride_v;
    ck::index_t nhead_stride_bias;
    ck::index_t nhead_stride_randval;
    ck::index_t nhead_stride_lse;
    ck::index_t nhead_stride_o;
    ck::index_t batch_stride_q;
    ck::index_t batch_stride_k;
    ck::index_t batch_stride_v;
    ck::index_t batch_stride_bias;
    ck::index_t batch_stride_randval;
    ck::index_t batch_stride_lse;
    ck::index_t batch_stride_o;
    ck::index_t mask_y;
    ck::index_t mask_x;
    float descale_qk;
    float descale_sv;
    float p_drop;
    bool s_randval;
    std::tuple<uint64_t, uint64_t> drop_seed_offset;
};

template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.bias_ptr,
                                         args.rand_val_ptr,
                                         args.lse_ptr,
                                         args.o_ptr,
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
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_bias,
                                         args.nhead_stride_randval,
                                         args.nhead_stride_lse,
                                         args.nhead_stride_o,
                                         args.mask_y,
                                         args.mask_x,
                                         args.descale_qk,
                                         args.descale_sv,
                                         args.p_drop,
                                         args.s_randval,
                                         args.drop_seed_offset);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.bias_ptr,
                                         args.rand_val_ptr,
                                         args.lse_ptr,
                                         args.o_ptr,
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
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_bias,
                                         args.nhead_stride_randval,
                                         args.nhead_stride_lse,
                                         args.nhead_stride_o,
                                         args.batch_stride_q,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.batch_stride_bias,
                                         args.batch_stride_randval,
                                         args.batch_stride_lse,
                                         args.batch_stride_o,
                                         args.mask_y,
                                         args.mask_x,
                                         args.descale_qk,
                                         args.descale_sv,
                                         args.p_drop,
                                         args.s_randval,
                                         args.drop_seed_offset);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck::index_t kM0_,
          ck::index_t kN0_,
          ck::index_t kK0_,
          ck::index_t kN1_,
          ck::index_t kK1_,
          ck::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          typename FmhaMask_,
          bool kHasBias_,
          bool kStoreLse_,
          bool kHasDropout_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_>
struct fmha_fwd_traits_
{
    static constexpr ck::index_t HDim           = HDim_;
    using DataType                              = ck::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode          = kIsGroupMode_;
    static constexpr ck::index_t kM0            = kM0_;
    static constexpr ck::index_t kN0            = kN0_;
    static constexpr ck::index_t kK0            = kK0_;
    static constexpr ck::index_t kN1            = kN1_;
    static constexpr ck::index_t kK1            = kK1_;
    static constexpr ck::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor    = kIsVLayoutRowMajor_;
    using FmhaMask                              = ck::remove_cvref_t<FmhaMask_>;
    static constexpr bool kHasBias              = kHasBias_;
    static constexpr bool kStoreLse             = kStoreLse_;
    static constexpr bool kHasDropout           = kHasDropout_;
    static constexpr bool kPadS                 = kPadS_;
    static constexpr bool kPadSK                = kPadSK_;
    static constexpr bool kPadD                 = kPadD_;
    static constexpr bool kPadDv                = kPadDv_;
};

template <typename Traits_>
float fmha_fwd_(const StreamConfig&, fmha_fwd_args);

// This is the public API, will be generated by script
struct fmha_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    mask_enum mask_type;
    bool has_bias;
    bool has_lse;
    bool has_dropout;
    // TODO: padding check is inside this api
};
float fmha_fwd(fmha_fwd_traits, fmha_fwd_args, const StreamConfig&);
