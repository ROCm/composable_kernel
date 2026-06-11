// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// This header is designed to be embedded and used at RTC compilation time.

#include <cmath>
#include <cstdint>
#include <cassert>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

namespace ck_tile {

enum class FmhaPipelineTag
{
    QR,             // BlockFmhaPipelineQRKSVS
    QR_ASYNC,       // BlockFmhaPipelineQRKSVSAsync
    QR_ASYNC_TRLOAD // BlockFmhaPipelineQRKSVSAsyncTrload
};

template <typename DataType_,
          // Block tile
          index_t kBM0,
          index_t kBN0,
          index_t kBK0,
          index_t kBN1,
          index_t kBK1,
          index_t kBK0Max,
          // Gemm0 block warps
          index_t kRM0,
          index_t kRN0,
          index_t kRK0,
          // Gemm1 block warps
          index_t kRM1,
          index_t kRN1,
          index_t kRK1,
          // Gemm0 warp tile
          index_t kWM0,
          index_t kWN0,
          index_t kWK0,
          // Gemm1 warp tile
          index_t kWM1,
          index_t kWN1,
          index_t kWK1,
          //
          bool kIsCausal,
          bool kIsVRowMajor,
          bool kHasBias,
          //
          bool kPadM,
          bool kPadN,
          bool kPadK,
          bool kPadO,
          //
          FmhaPipelineTag kPipelineTag>
struct FmhaFwdWrapper
{
    using BlockTile = sequence<kBM0, kBN0, kBK0, kBN1, kBK1, kBK0Max>;

    using Gemm0BlockWarps = sequence<kRM0, kRN0, kRK0>;
    using Gemm0WarpTile   = sequence<kWM0, kWN0, kWK0>;
    using Gemm1BlockWarps = sequence<kRM1, kRN1, kRK1>;
    using Gemm1WarpTile   = sequence<kWM1, kWN1, kWK1>;

    using FmhaShape = TileFmhaShape<BlockTile,
                                    Gemm0BlockWarps,
                                    Gemm0WarpTile,
                                    Gemm1BlockWarps,
                                    Gemm1WarpTile,
                                    kIsVRowMajor>;

    static constexpr auto BiasEnum =
        kHasBias ? BlockAttentionBiasEnum::ELEMENTWISE_BIAS : BlockAttentionBiasEnum::NO_BIAS;

    using FmhaTraits = TileFmhaTraits<kPadM, // kPadSeqLenQ
                                      kPadN, // kPadSeqLenK
                                      kPadK, // kPadHeadDimQ
                                      kPadO, // kPadHeadDimV
                                      false, // kHasLogitsSoftCap
                                      BiasEnum,
                                      false, // kHasBiasGrad
                                      false, // kStoreLSE
                                      false, // kHasDropout
                                      BlockAttentionQuantScaleEnum::NO_SCALE,
                                      -1,     // kBlockPerCu
                                      false,  // kSkipMinSeqlenQ
                                      false>; // kHasSink

    using FmhaMask = std::conditional_t<kIsCausal,
                                        SimplifiedGenericAttentionMask<true>,
                                        SimplifiedGenericAttentionMask<false>>;

    static constexpr bool kUseTrLoad = (kPipelineTag == FmhaPipelineTag::QR_ASYNC_TRLOAD);

    using PipelineProblem =
        BlockFmhaPipelineProblem<DataType_,      // Q type
                                 DataType_,      // K type
                                 DataType_,      // V type
                                 float,          // Sacc type
                                 float,          // SMPLCompute type
                                 DataType_,      // Bias type
                                 unsigned short, // RandVal type (unused)
                                 float,          // LSE type (unused)
                                 DataType_,      // P type
                                 float,          // Oacc type
                                 DataType_,      // O type
                                 FmhaShape,
                                 false, // mode (false = batch mode)
                                 ComposedAttention<false, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                 FmhaMask,
                                 kUseTrLoad,
                                 FmhaTraits>;

    using Pipeline =
        std::conditional_t<kPipelineTag == FmhaPipelineTag::QR_ASYNC_TRLOAD,
                           BlockFmhaPipelineQRKSVSAsyncTrload<PipelineProblem>,
                           std::conditional_t<kPipelineTag == FmhaPipelineTag::QR_ASYNC,
                                              BlockFmhaPipelineQRKSVSAsync<PipelineProblem>,
                                              BlockFmhaPipelineQRKSVS<PipelineProblem>>>;

    using Epilogue = Default2DEpilogue<Default2DEpilogueProblem<float, DataType_, kPadM, kPadO>>;

    using Kernel = FmhaFwdKernel<Pipeline, Epilogue>;

    // Innermost dimension is always contiguous (stride=1):
    //
    // K is stored as [batch, nhead, N, K] (not transposed).
    // The kernel internally handles the transpose for Q @ K^T.
    //
    // Q: [batch, nhead, M, K]
    // K: [batch, nhead, N, K]
    // V: [batch, nhead, N, O] (rowmajor) or [batch, nhead, O, N] (colmajor)
    // O: [batch, nhead, M, O]
    // Bias: [batch, nhead, M, N]
    struct Descriptor
    {
        index_t batch, nhead, M, K;
        index_t q_stride_batch, q_stride_nhead, q_stride_m;

        index_t N;
        index_t k_stride_batch, k_stride_nhead, k_stride_n;

        index_t O;
        index_t v_stride_batch, v_stride_nhead, v_stride_n;

        index_t o_stride_batch, o_stride_nhead, o_stride_m;

        index_t bias_stride_batch, bias_stride_nhead, bias_stride_m;

        // Only reflects compile time arch availability,
        // does not perform runtime descriptor validation.
        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return Kernel::kIsAvailable; }
    };

    // Each tensor is specified as (batch, nhead, dim0, dim1) and (stride0, stride1, stride2)
    // Innermost stride is always 1 and not passed.
    template <typename QDims,
              typename QStrides,
              typename KDims,
              typename KStrides,
              typename VDims,
              typename VStrides,
              typename ODims,
              typename OStrides,
              typename BiasDims,
              typename BiasStrides>
    CK_TILE_HOST_DEVICE static constexpr auto make_descriptor(QDims q_dims,
                                                              QStrides q_strides,
                                                              KDims k_dims,
                                                              KStrides k_strides,
                                                              VDims v_dims,
                                                              VStrides v_strides,
                                                              ODims o_dims,
                                                              OStrides o_strides,
                                                              BiasDims bias_dims,
                                                              BiasStrides bias_strides)
    {
        return Descriptor{q_dims[number<0>{}],
                          q_dims[number<1>{}],
                          q_dims[number<2>{}],
                          q_dims[number<3>{}],
                          q_strides[number<0>{}],
                          q_strides[number<1>{}],
                          q_strides[number<2>{}],
                          //
                          k_dims[number<2>{}],
                          k_strides[number<0>{}],
                          k_strides[number<1>{}],
                          k_strides[number<2>{}],
                          //
                          v_dims[number<3>{}],
                          v_strides[number<0>{}],
                          v_strides[number<1>{}],
                          v_strides[number<2>{}],
                          //
                          o_strides[number<0>{}],
                          o_strides[number<1>{}],
                          o_strides[number<2>{}],
                          //
                          bias_strides[number<0>{}],
                          bias_strides[number<1>{}],
                          bias_strides[number<2>{}]};
    }

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   float scale_s,
                                   const DataType_* q_ptr,
                                   const DataType_* k_ptr,
                                   const DataType_* v_ptr,
                                   const DataType_* bias_ptr,
                                   DataType_* o_ptr)
    {
        using Kargs = typename Kernel::Kargs;
        Kargs kargs{};

        kargs.q_ptr    = q_ptr;
        kargs.k_ptr    = k_ptr;
        kargs.v_ptr    = v_ptr;
        kargs.o_ptr    = o_ptr;
        kargs.sink_ptr = nullptr;

        kargs.seqlen_q = desc.M;
        kargs.seqlen_k = desc.N;
        kargs.hdim_q   = desc.K;
        kargs.hdim_v   = desc.O;

        kargs.num_head_q     = desc.nhead;
        kargs.nhead_ratio_qk = 1; // nhead_q == nhead_k

        kargs.scale_s = scale_s;

        kargs.stride_q = desc.q_stride_m;
        kargs.stride_k = desc.k_stride_n;
        kargs.stride_v = desc.v_stride_n;
        kargs.stride_o = desc.o_stride_m;

        kargs.nhead_stride_q = desc.q_stride_nhead;
        kargs.nhead_stride_k = desc.k_stride_nhead;
        kargs.nhead_stride_v = desc.v_stride_nhead;
        kargs.nhead_stride_o = desc.o_stride_nhead;

        if constexpr(kHasBias)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = desc.bias_stride_m;
            kargs.nhead_stride_bias = desc.bias_stride_nhead;
            kargs.batch_stride_bias = desc.bias_stride_batch;
        }

        if constexpr(kIsCausal)
        {
            kargs.window_size_left  = -1;
            kargs.window_size_right = 0;
            kargs.sink_size         = 0;
            kargs.mask_type         = GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT;
        }

        kargs.batch_stride_q = desc.q_stride_batch;
        kargs.batch_stride_k = desc.k_stride_batch;
        kargs.batch_stride_v = desc.v_stride_batch;
        kargs.batch_stride_o = desc.o_stride_batch;

        kargs.cu_seqlen_q_ptr = nullptr;
        kargs.cu_seqlen_k_ptr = nullptr;

        Kernel{}(kargs);
    }
};

} // namespace ck_tile
