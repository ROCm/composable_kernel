// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/common.hpp"
#include "mask.hpp"

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<ck_tile::half_t>
{
    using QDataType           = ck_tile::half_t;
    using KDataType           = ck_tile::half_t;
    using VDataType           = ck_tile::half_t;
    using BiasDataType        = ck_tile::half_t;
    using LSEDataType         = float;      // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float;      // data type for first gemm accumulation
    using SMPLComputeDataType = float;      // data type for reduction, softmax
    using PDataType           = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType        = float;      // data type for second gemm accumulation
    using ODataType           = ck_tile::half_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::bhalf_t>
{
    using QDataType           = ck_tile::bhalf_t;
    using KDataType           = ck_tile::bhalf_t;
    using VDataType           = ck_tile::bhalf_t;
    using BiasDataType        = ck_tile::bhalf_t;
    using LSEDataType         = float;       // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float;       // data type for first gemm accumulation
    using SMPLComputeDataType = float;       // data type for reduction, softmax
    using PDataType           = ck_tile::bhalf_t; // data type for A matrix of second gemm
    using OaccDataType        = float;       // data type for second gemm accumulation
    using ODataType           = ck_tile::bhalf_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::fp8_t>
{
    using QDataType           = ck_tile::fp8_t;
    using KDataType           = ck_tile::fp8_t;
    using VDataType           = ck_tile::fp8_t;
    using BiasDataType        = float;    // TODO: fix me
    using LSEDataType         = float;    // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float;    // data type for first gemm accumulation
    using SMPLComputeDataType = float;    // data type for reduction, softmax
    using PDataType           = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType        = float;    // data type for second gemm accumulation
    using ODataType           = ck_tile::fp8_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::bf8_t>
{
    using QDataType           = ck_tile::bf8_t;
    using KDataType           = ck_tile::bf8_t;
    using VDataType           = ck_tile::bf8_t;
    using BiasDataType        = ck_tile::bf8_t;
    using LSEDataType         = float;     // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float;     // data type for first gemm accumulation
    using SMPLComputeDataType = float;     // data type for reduction, softmax
    using PDataType           = ck_tile::bf8_t; // data type for A matrix of second gemm
    using OaccDataType        = float;     // data type for second gemm accumulation
    using ODataType           = ck_tile::bf8_t;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
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
                                     ck_tile::index_t batch,
                                     ck_tile::index_t nhead,
                                     ck_tile::index_t nhead_k,
                                     ck_tile::index_t seqlen_q,
                                     ck_tile::index_t seqlen_k,
                                     ck_tile::index_t hdim_q,
                                     ck_tile::index_t hdim_v,
                                     ck_tile::index_t max_seqlen_q,
                                     float scale,
                                     float descale_qk,
                                     float descale_sv,
                                     bool i_perm,
                                     bool o_perm,
                                     ck_tile::index_t mask_y,
                                     ck_tile::index_t mask_x)
{
    constexpr bool is_v_rowmajor =
        ck_tile::is_same_v<typename FmhaKernel::VLayout, ck_tile::tensor_layout::gemm::RowMajor>;

    assert(nhead % nhead_k == 0);
    /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' & 'nhead_stride_bias'
    ///       are 0.
    // setup stride_* arguments
    const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
    const ck_tile::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
    const ck_tile::index_t stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? hdim_v : nhead_k * hdim_v;
        else
            return i_perm ? seqlen_k : nhead_k * seqlen_k;
    }();
    const ck_tile::index_t stride_bias = (i_perm ? seqlen_k : 1 * seqlen_k);
    const ck_tile::index_t stride_o    = (o_perm ? hdim_v : nhead * hdim_v);
    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = (i_perm ? seqlen_q * hdim_q : hdim_q);
    const ck_tile::index_t nhead_stride_k = (i_perm ? seqlen_k * hdim_q : hdim_q);
    const ck_tile::index_t nhead_stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? seqlen_k * hdim_v : hdim_v;
        else
            return i_perm ? hdim_v * seqlen_k : seqlen_k;
    }();
    const ck_tile::index_t nhead_stride_bias = (i_perm ? 0 * seqlen_q * seqlen_k : 0 * seqlen_k);
    const ck_tile::index_t nhead_stride_lse  = (seqlen_q * 1);
    const ck_tile::index_t nhead_stride_o    = (o_perm ? seqlen_q * hdim_v : hdim_v);
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q    = (nhead * seqlen_q * hdim_q);
    const ck_tile::index_t batch_stride_k    = (nhead_k * seqlen_k * hdim_q);
    const ck_tile::index_t batch_stride_v    = (nhead_k * hdim_v * seqlen_k);
    const ck_tile::index_t batch_stride_bias = (0 * nhead * seqlen_q * seqlen_k);
    const ck_tile::index_t batch_stride_lse  = (nhead * seqlen_q * 1);
    const ck_tile::index_t batch_stride_o    = (nhead * seqlen_q * hdim_v);

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
    return ck_tile::make_tuple(kargs, grids);
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
    ck_tile::index_t batch;
    ck_tile::index_t nhead;
    ck_tile::index_t nhead_k;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t max_seqlen_q;
    float scale;
    float descale_qk;
    float descale_sv;
    bool i_perm;
    bool o_perm;
    ck_tile::index_t mask_y;
    ck_tile::index_t mask_x;
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

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          typename FmhaMask_,
          bool kHasBias_,
          bool kStoreLse_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_>
struct fmha_fwd_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                              = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode          = kIsGroupMode_;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor    = kIsVLayoutRowMajor_;
    using FmhaMask                              = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr bool kHasBias              = kHasBias_;
    static constexpr bool kStoreLse             = kStoreLse_;
    static constexpr bool kPadS                 = kPadS_;
    static constexpr bool kPadSK                = kPadSK_;
    static constexpr bool kPadD                 = kPadD_;
    static constexpr bool kPadDv                = kPadDv_;
};

template <typename Traits_>
float fmha_fwd_(const stream_config&, fmha_fwd_args);

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
    // TODO: padding check is inside this api
};
float fmha_fwd(fmha_fwd_traits, fmha_fwd_args, const stream_config&);
