// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"

#include "flash_attention_fwd_impl.hpp"

// S[M0, N0] = Q[M0, K0] * K[N0, K0]
// P[M0, N0] = Softmax(S[M0, N0])
// O[M0, N1] = P[M0, N0] * V[N1, N0]
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename ODataType,
          ck::index_t kBlockSize,
          ck::index_t kHeadDim,
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock,
          ck::index_t kK1PerBlock>
struct FlashAttentionFwd
{
    __device__ void operator()(const QDataType* q_ptr,
                               const KDataType* k_ptr,
                               const VDataType* v_ptr,
                               ODataType* o_ptr,
                               const ck::index_t M0,
                               const ck::index_t N0,
                               const ck::index_t K0,
                               const ck::index_t N1,
                               const ck::index_t /* Batch */,
                               const ck::index_t StrideQ,
                               const ck::index_t StrideK,
                               const ck::index_t StrideV,
                               const ck::index_t StrideO,
                               const ck::index_t BatchStrideQ,
                               const ck::index_t BatchStrideK,
                               const ck::index_t BatchStrideV,
                               const ck::index_t BatchStrideO) const
    {
        using namespace ck;

        // divide problem
        const index_t num_tile_m0 = M0 / kM0PerBlock;
        const index_t num_tile_n1 = N1 / kN1PerBlock;

        const index_t id_block = get_block_id();

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;

            return ck::make_tuple(quotient, modulus);
        };

        const auto [itmp, id_tile_n]          = f(id_block, num_tile_n1);
        const auto [id_tile_batch, id_tile_m] = f(itmp, num_tile_m0);

        const index_t iBatch = __builtin_amdgcn_readfirstlane(id_tile_batch);
        const index_t iM0    = __builtin_amdgcn_readfirstlane(id_tile_m * kM0PerBlock);
        const index_t iN1    = __builtin_amdgcn_readfirstlane(id_tile_n * kN1PerBlock);

        const auto kernel_impl = FlashAttentionFwdImpl<QDataType,
                                                       KDataType,
                                                       VDataType,
                                                       SaccDataType,
                                                       SMPLComputeDataType,
                                                       PDataType,
                                                       OaccDataType,
                                                       ODataType,
                                                       kBlockSize,
                                                       kHeadDim,
                                                       kM0PerBlock,
                                                       kN0PerBlock,
                                                       kK0PerBlock,
                                                       kN1PerBlock,
                                                       kK1PerBlock>{};

        kernel_impl(q_ptr + iBatch * BatchStrideQ,
                    k_ptr + iBatch * BatchStrideK,
                    v_ptr + iBatch * BatchStrideV,
                    o_ptr + iBatch * BatchStrideO,
                    M0,
                    N0,
                    K0,
                    N1,
                    StrideQ,
                    StrideK,
                    StrideV,
                    StrideO,
                    iM0,
                    iN1);
    }
};
