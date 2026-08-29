// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha.hpp"

#include "01_fmha/mask.hpp"

#include <type_traits>
#include <utility>
#include <variant>

namespace ck_tile {
inline bool is_load_tr_supported() { return is_gfx95_supported(); }
} // namespace ck_tile

struct FmhaSparseFwdFp16
{
};

struct FmhaSparseFwdBf16
{
};

template <typename DataType>
struct FmhaSparseFwdTypeConfig;

template <>
struct FmhaSparseFwdTypeConfig<FmhaSparseFwdFp16>
{
    using QDataType           = ck_tile::half_t;
    using KDataType           = ck_tile::half_t;
    using VDataType           = ck_tile::half_t;
    using SaccDataType        = float;           // data type for first gemm accumulation
    using SMPLComputeDataType = float;           // data type for reduction, softmax
    using PDataType           = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType        = float;           // data type for second gemm accumulation
    using ODataType           = ck_tile::half_t;
    // Note: The following types are required by BlockFmhaPipelineProblem but not used
    // by sparse attention (bias, dropout, LSE are not supported).
    using BiasDataType          = ck_tile::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;
};

template <>
struct FmhaSparseFwdTypeConfig<FmhaSparseFwdBf16>
{
    using QDataType           = ck_tile::bf16_t;
    using KDataType           = ck_tile::bf16_t;
    using VDataType           = ck_tile::bf16_t;
    using SaccDataType        = float;           // data type for first gemm accumulation
    using SMPLComputeDataType = float;           // data type for reduction, softmax
    using PDataType           = ck_tile::bf16_t; // data type for A matrix of second gemm
    using OaccDataType        = float;           // data type for second gemm accumulation
    using ODataType           = ck_tile::bf16_t;
    // Note: The following types are required by BlockFmhaPipelineProblem but not used
    // by sparse attention (bias, dropout, LSE are not supported).
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// jenga
struct fmha_jenga_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* block_relation_onehot_ptr; // one-hot block map [B,H,Q_blk,K_blk], 1=active
    void* o_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;

    // Dropout is not supported for sparse attention; keep args minimal.
};

// vsa
struct fmha_vsa_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* lut_ptr; // delta-encoded K-block indices per Q-block, int32 [B,H,Q_blk,K_blk]
    const void* valid_block_num_ptr; // valid K-block count per Q-block, int32 [B,H,Q_blk]
    void* o_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;

    // Dropout is not supported for sparse attention; keep args minimal.
};

template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_jenga_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = FmhaKernel::MakeKargs(args.q_ptr,
                                       args.k_ptr,
                                       args.v_ptr,
                                       args.block_relation_onehot_ptr,
                                       args.o_ptr,
                                       args.seqlen_q,
                                       args.seqlen_k,
                                       args.hdim_q,
                                       args.hdim_v,
                                       args.nhead_q,
                                       args.nhead_q / args.nhead_k,
                                       args.scale_s,
                                       args.stride_q,
                                       args.stride_k,
                                       args.stride_v,
                                       args.stride_o,
                                       args.nhead_stride_q,
                                       args.nhead_stride_k,
                                       args.nhead_stride_v,
                                       args.nhead_stride_o,
                                       args.batch_stride_q,
                                       args.batch_stride_k,
                                       args.batch_stride_v,
                                       args.batch_stride_o,
                                       args.window_size_left,
                                       args.window_size_right,
                                       args.mask_type);

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_vsa_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = FmhaKernel::MakeKargs(args.q_ptr,
                                       args.k_ptr,
                                       args.v_ptr,
                                       args.lut_ptr,
                                       args.valid_block_num_ptr,
                                       args.o_ptr,
                                       args.seqlen_q,
                                       args.seqlen_k,
                                       args.hdim_q,
                                       args.hdim_v,
                                       args.nhead_q,
                                       args.nhead_q / args.nhead_k,
                                       args.scale_s,
                                       args.stride_q,
                                       args.stride_k,
                                       args.stride_v,
                                       args.stride_o,
                                       args.nhead_stride_q,
                                       args.nhead_stride_k,
                                       args.nhead_stride_v,
                                       args.nhead_stride_o,
                                       args.batch_stride_q,
                                       args.batch_stride_k,
                                       args.batch_stride_v,
                                       args.batch_stride_o,
                                       args.window_size_left,
                                       args.window_size_right,
                                       args.mask_type);

    dim3 grids = FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v);
    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_>
struct fmha_jenga_fwd_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kUseTrLoad                 = kUseTrLoad_;
};

struct fmha_jenga_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_v_rowmajor;
    mask_enum mask_type;
    // TODO: padding check is inside this api
};

float fmha_jenga_fwd(fmha_jenga_fwd_traits, fmha_jenga_fwd_args, const ck_tile::stream_config&);

template <typename Traits_>
float fmha_jenga_fwd_(const ck_tile::stream_config&, fmha_jenga_fwd_args);

float fmha_jenga_fwd(fmha_jenga_fwd_args, const ck_tile::stream_config&);

// VSA uses the same traits structure as Jenga; aliases for clarity
template <ck_tile::index_t HDim_,
          typename DataType_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_>
using fmha_vsa_fwd_traits_ = fmha_jenga_fwd_traits_<HDim_,
                                                    DataType_,
                                                    kM0_,
                                                    kN0_,
                                                    kK0_,
                                                    kN1_,
                                                    kK1_,
                                                    kK0BlockLength_,
                                                    kIsVLayoutRowMajor_,
                                                    FmhaPipelineEnum_,
                                                    kHasLogitsSoftCap_,
                                                    FmhaMask_,
                                                    kPadS_,
                                                    kPadSK_,
                                                    kPadD_,
                                                    kPadDv_,
                                                    kUseTrLoad_>;

using fmha_vsa_fwd_traits = fmha_jenga_fwd_traits;

float fmha_vsa_fwd(fmha_vsa_fwd_traits, fmha_vsa_fwd_args, const ck_tile::stream_config&);

template <typename Traits_>
float fmha_vsa_fwd_(const ck_tile::stream_config&, fmha_vsa_fwd_args);

float fmha_vsa_fwd(fmha_vsa_fwd_args, const ck_tile::stream_config&);
