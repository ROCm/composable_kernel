// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "mask.hpp"
#include <type_traits>

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<ck_tile::half_t>
{
    using QDataType           = ck_tile::half_t;
    using KDataType           = ck_tile::half_t;
    using VDataType           = ck_tile::half_t;
    using BiasDataType        = ck_tile::half_t;
    using LSEDataType         = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float; // data type for first gemm accumulation
    using SMPLComputeDataType = float; // data type for reduction, softmax
    using PDataType           = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType        = float;           // data type for second gemm accumulation
    using ODataType           = ck_tile::half_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::bf16_t>
{
    using QDataType           = ck_tile::bf16_t;
    using KDataType           = ck_tile::bf16_t;
    using VDataType           = ck_tile::bf16_t;
    using BiasDataType        = ck_tile::bf16_t;
    using LSEDataType         = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float; // data type for first gemm accumulation
    using SMPLComputeDataType = float; // data type for reduction, softmax
    using PDataType           = ck_tile::bf16_t; // data type for A matrix of second gemm
    using OaccDataType        = float;           // data type for second gemm accumulation
    using ODataType           = ck_tile::bf16_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::fp8_t>
{
    using QDataType           = ck_tile::fp8_t;
    using KDataType           = ck_tile::fp8_t;
    using VDataType           = ck_tile::fp8_t;
    using BiasDataType        = float;
    using LSEDataType         = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float; // data type for first gemm accumulation
    using SMPLComputeDataType = float; // data type for reduction, softmax
    using PDataType           = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType        = float;          // data type for second gemm accumulation
    using ODataType           = ck_tile::fp8_t;
};

struct FmhaDefaultElementFunctions
{
    // using QElementFunction        = ck_tile::identity;
    // using KElementFunction        = ck_tile::identity;
    // using VElementFunction        = ck_tile::identity;
    // using BiasElementFunction     = ck_tile::identity;
    // using LSEElementFunction      = ck_tile::identity;
    // using SAccElementFunction     = ck_tile::identity;
    using PComputeElementFunction = ck_tile::identity;
    using OAccElementFunction     = ck_tile::identity;
};

struct FmhaF8StaticQuantizationElementFunctions
{
    // using QElementFunction        = ck_tile::identity;
    // using KElementFunction        = ck_tile::identity;
    // using VElementFunction        = ck_tile::identity;
    // using BiasElementFunction     = ck_tile::identity;
    // using LSEElementFunction      = ck_tile::identity;
    // using SAccElementFunction     = ck_tile::identity;
    using PComputeElementFunction = ck_tile::scale;
    using OAccElementFunction     = ck_tile::composer<ck_tile::saturate_f8, ck_tile::scale>;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::bf8_t>
{
    using QDataType           = ck_tile::bf8_t;
    using KDataType           = ck_tile::bf8_t;
    using VDataType           = ck_tile::bf8_t;
    using BiasDataType        = ck_tile::bf8_t;
    using LSEDataType         = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType        = float; // data type for first gemm accumulation
    using SMPLComputeDataType = float; // data type for reduction, softmax
    using PDataType           = ck_tile::bf8_t; // data type for A matrix of second gemm
    using OaccDataType        = float;          // data type for second gemm accumulation
    using ODataType           = ck_tile::bf8_t;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
template <typename ElementFunctions>
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
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    float scale;
    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    // typename ElementFunctions::QElementFunction q_element_func;
    // typename ElementFunctions::KElementFunction k_element_func;
    // typename ElementFunctions::VElementFunction v_element_func;
    // typename ElementFunctions::BiasElementFunction bias_element_func;
    // typename ElementFunctions::LSEElementFunction lse_element_func;
    // typename ElementFunctions::SAccElementFunction s_acc_element_func;
    typename ElementFunctions::PComputeElementFunction p_compute_element_func;
    typename ElementFunctions::OAccElementFunction o_acc_element_func;
};

template <typename FmhaKernel, typename FmhaFwdArgs>
auto fmha_fwd_create_kargs_and_grids(FmhaFwdArgs args)
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
                                         args.lse_ptr,
                                         args.o_ptr,
                                         args.seqstart_q_ptr,
                                         args.seqstart_k_ptr,
                                         args.seqlen_k_ptr,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q / args.nhead_k,
                                         args.scale,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_bias,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_bias,
                                         args.nhead_stride_lse,
                                         args.nhead_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         //  args.q_element_func,
                                         //  args.k_element_func,
                                         //  args.v_element_func,
                                         //  args.bias_element_func,
                                         //  args.lse_element_func,
                                         //  args.s_acc_element_func,
                                         args.p_compute_element_func,
                                         args.o_acc_element_func);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargs(args.q_ptr,
                                         args.k_ptr,
                                         args.v_ptr,
                                         args.bias_ptr,
                                         args.lse_ptr,
                                         args.o_ptr,
                                         args.seqlen_q,
                                         args.seqlen_k,
                                         args.hdim_q,
                                         args.hdim_v,
                                         args.nhead_q / args.nhead_k,
                                         args.scale,
                                         args.stride_q,
                                         args.stride_k,
                                         args.stride_v,
                                         args.stride_bias,
                                         args.stride_o,
                                         args.nhead_stride_q,
                                         args.nhead_stride_k,
                                         args.nhead_stride_v,
                                         args.nhead_stride_bias,
                                         args.nhead_stride_lse,
                                         args.nhead_stride_o,
                                         args.batch_stride_q,
                                         args.batch_stride_k,
                                         args.batch_stride_v,
                                         args.batch_stride_bias,
                                         args.batch_stride_lse,
                                         args.batch_stride_o,
                                         args.window_size_left,
                                         args.window_size_right,
                                         args.mask_type,
                                         //  args.q_element_func,
                                         //  args.k_element_func,
                                         //  args.v_element_func,
                                         //  args.bias_element_func,
                                         //  args.lse_element_func,
                                         //  args.s_acc_element_func,
                                         args.p_compute_element_func,
                                         args.o_acc_element_func);
        }
    }();

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck_tile::make_tuple(kargs, grids);
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
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
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
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr bool kHasBias                   = kHasBias_;
    static constexpr bool kStoreLse                  = kStoreLse_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
};

template <typename Traits_, typename FmhaFwdArgs_>
float fmha_fwd_(const ck_tile::stream_config&, FmhaFwdArgs_);

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

template <typename FmhaFwdArgs_>
float fmha_fwd(fmha_fwd_traits, FmhaFwdArgs_, const ck_tile::stream_config&);
