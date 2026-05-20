// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/sageattn.hpp"

#include "mask.hpp"
#include "quant.hpp"

#include <type_traits>
#include <utility>
#include <variant>

// SageAttention data type configs (must match codegen FWD_DTYPE_MAP + SageAttentionFwdTypeConfig)
struct SageAttentionFwdFp16
{
};

struct SageAttentionFwdBf16
{
};

struct SageAttentionFwdFp8Bf16
{
};

struct SageAttentionFwdI8Fp8Bf16
{
};

struct SageAttentionFwdI4Fp8Bf16
{
};

template <typename DataType>
struct SageAttentionFwdTypeConfig;

// fp16/bf16 are not Sage product dtypes; bf16 is intentionally kept in tile_example_sageattn_fwd
// for pipeline validation with qscale=n (no quant).
template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdFp16>
{
    using QDataType           = ck_tile::half_t;
    using KDataType           = ck_tile::half_t;
    using VDataType           = ck_tile::half_t;
    using ScaleType           = float;           // scale type for quantized inputs
    using SaccDataType        = float;           // data type for first gemm accumulation
    using SMPLComputeDataType = float;           // data type for reduction, softmax
    using PDataType           = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType        = float;           // data type for second gemm accumulation
    using ODataType           = ck_tile::half_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdBf16>
{
    using QDataType           = ck_tile::bf16_t;
    using KDataType           = ck_tile::bf16_t;
    using VDataType           = ck_tile::bf16_t;
    using ScaleType           = float;           // scale type for quantized inputs
    using SaccDataType        = float;           // data type for first gemm accumulation
    using SMPLComputeDataType = float;           // data type for reduction, softmax
    using PDataType           = ck_tile::bf16_t; // data type for A matrix of second gemm
    using OaccDataType        = float;           // data type for second gemm accumulation
    using ODataType           = ck_tile::bf16_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdFp8Bf16>
{
    using QDataType           = ck_tile::fp8_t;
    using KDataType           = ck_tile::fp8_t;
    using VDataType           = ck_tile::fp8_t;
    using ScaleType           = float;          // scale type for quantized inputs
    using SaccDataType        = float;          // data type for first gemm accumulation
    using SMPLComputeDataType = float;          // data type for reduction, softmax
    using PDataType           = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType        = float;          // data type for second gemm accumulation
    using ODataType           = ck_tile::bf16_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdI8Fp8Bf16>
{
    using QDataType           = ck_tile::int8_t;
    using KDataType           = ck_tile::int8_t;
    using VDataType           = ck_tile::fp8_t;
    using ScaleType           = float;          // scale type for Q and K
    using SaccDataType        = float;          // Keep as float for softmax computation
    using SMPLComputeDataType = float;          // data type for reduction, softmax
    using PDataType           = ck_tile::fp8_t; // P in FP8 for 2nd gemm
    using OaccDataType        = float;          // data type for second gemm accumulation
    using ODataType           = ck_tile::bf16_t;
};

template <>
struct SageAttentionFwdTypeConfig<SageAttentionFwdI4Fp8Bf16>
{
    using QDataType           = ck_tile::pk_int4_t;
    using KDataType           = ck_tile::pk_int4_t;
    using VDataType           = ck_tile::fp8_t;
    using ScaleType           = float;
    using SaccDataType        = float;
    using SMPLComputeDataType = float;
    using PDataType           = ck_tile::fp8_t;
    using OaccDataType        = float;
    using ODataType           = ck_tile::bf16_t;
};

struct SageAttnMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct sageattn_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* q_descale_ptr;
    const void* k_descale_ptr;
    const void* v_descale_ptr;
    void* o_ptr;

    // Usage notes for sequence length pointer parameters:
    //
    // [Note: Define "Group mode" vs "Batch mode" here if possible, e.g., "Group mode handles
    // MQA/GQA..."]
    //
    // With padding:
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical (including padding) sequence
    //     lengths. [array size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr: Records logical (excluding padding) length for each
    //     sequence. [array size: batch]
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqlen_q_ptr (per-sequence) and cu_seqlen_q_ptr (cumulative logical) are mutually
    //     exclusive. Use one set, not both.
    //
    //   Batch mode:
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqstart_* and seqlen_* pointers must be nullptr.
    //
    // Without padding:
    //   (Note: Physical length equals logical length)
    //
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical sequence lengths. [array
    //     size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr and cu_seqlen_q_ptr/cu_seqlen_k_ptr must be nullptr.
    //
    //   Batch mode:
    //     - All sequence length pointers (seqstart_*, seqlen_*, cu_seqlen_*) must be nullptr.
    //
    const void* seqstart_q_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqstart_k_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqlen_q_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* seqlen_k_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* cu_seqlen_q_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* cu_seqlen_k_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)

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
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;

    // BLOCKSCALE parameters
    ck_tile::index_t nhead_stride_q_descale = 0;
    ck_tile::index_t nhead_stride_k_descale = 0;
    ck_tile::index_t nhead_stride_v_descale = 0;
    ck_tile::index_t batch_stride_q_descale = 0;
    ck_tile::index_t batch_stride_k_descale = 0;
    ck_tile::index_t batch_stride_v_descale = 0;
    ck_tile::index_t block_scale_size_q     = 0;
    ck_tile::index_t block_scale_size_k     = 0;
    const void* block_scale_seqstart_q_ptr  = nullptr;
    const void* block_scale_seqstart_k_ptr  = nullptr;
};

template <typename SageAttnKernel>
auto sageattn_fwd_create_kargs_and_grids(sageattn_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(SageAttnKernel::kIsGroupMode)
        {
            return SageAttnKernel::MakeKargs(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.q_descale_ptr,
                                             args.k_descale_ptr,
                                             args.v_descale_ptr,
                                             args.o_ptr,
                                             args.seqstart_q_ptr,
                                             args.seqstart_k_ptr,
                                             args.seqlen_q_ptr,
                                             args.seqlen_k_ptr,
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
                                             args.nhead_stride_q_descale,
                                             args.nhead_stride_k_descale,
                                             args.nhead_stride_v_descale,
                                             args.batch_stride_v_descale,
                                             args.block_scale_size_q,
                                             args.block_scale_size_k,
                                             args.block_scale_seqstart_q_ptr,
                                             args.block_scale_seqstart_k_ptr,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.mask_type,
                                             args.min_seqlen_q,
                                             args.cu_seqlen_q_ptr,
                                             args.cu_seqlen_k_ptr);
        }
        else
        { // create batch mode kernel arguments
            return SageAttnKernel::MakeKargs(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.q_descale_ptr,
                                             args.k_descale_ptr,
                                             args.v_descale_ptr,
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
                                             args.nhead_stride_q_descale,
                                             args.nhead_stride_k_descale,
                                             args.nhead_stride_v_descale,
                                             args.batch_stride_q,
                                             args.batch_stride_k,
                                             args.batch_stride_v,
                                             args.batch_stride_o,
                                             args.batch_stride_q_descale,
                                             args.batch_stride_k_descale,
                                             args.batch_stride_v_descale,
                                             args.block_scale_size_q,
                                             args.block_scale_size_k,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.mask_type,
                                             args.cu_seqlen_q_ptr,
                                             args.cu_seqlen_k_ptr);
        }
    }();

    if constexpr(SageAttnKernel::kIsGroupMode)
    {
        dim3 grids = SageAttnKernel::GridSize(
            args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, args.seqlen_k_ptr != nullptr);
        return ck_tile::make_tuple(kargs, grids);
    }
    else
    {
        dim3 grids = SageAttnKernel::GridSize(
            args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, false);
        return ck_tile::make_tuple(kargs, grids);
    }
}

// this is used to pattern-match internal kernel implementation, not to instantiate kernel
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
          ck_tile::BlockSageAttnPipelineEnum SageAttnPipelineEnum_,
          typename AttnMask_,
          ck_tile::BlockSageAttentionQuantScaleEnum QScaleEnum_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kSkipMinSeqlenQ_ = false>
struct sageattn_fwd_traits_
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
    static constexpr auto SageAttnPipelineEnum       = SageAttnPipelineEnum_;
    using AttnMask                                   = ck_tile::remove_cvref_t<AttnMask_>;
    static constexpr auto QScaleEnum                 = QScaleEnum_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kSkipMinSeqlenQ            = kSkipMinSeqlenQ_;
};

template <typename Traits_, typename Arch = void>
float sageattn_fwd_(const ck_tile::stream_config&, sageattn_fwd_args);

// This is the public API, will be generated by script
struct sageattn_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    mask_enum mask_type;
    quant_scale_enum qscale_type;
    bool skip_min_seqlen_q = false;
    // TODO: padding check is inside this api
};
float sageattn_fwd(sageattn_fwd_traits, sageattn_fwd_args, const ck_tile::stream_config&);
