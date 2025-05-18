// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "block_gemm_pipeline_problem.hpp"
#include "block_gemm_areg_bsmem_creg_v1.hpp"
#include "flash_attention_fwd_impl.hpp"

namespace ck_tile {

template <typename QDataType, typename KDataType, typename VDataType, typename ODataType>
struct FlashAttnArgs
{
    // Pointers to device buffers for Q, K, V, O
    QDataType* q_ptr;
    KDataType* k_ptr;
    VDataType* v_ptr;
    ODataType* o_ptr;

    // Problem sizes
    index_t M0;
    index_t N0;
    index_t K0;
    index_t N1;
    index_t Batch;

    // Strides within a batch
    index_t strideQ;
    index_t strideK;
    index_t strideV;
    index_t strideO;

    // Batch strides
    index_t batchStrideQ;
    index_t batchStrideK;
    index_t batchStrideV;
    index_t batchStrideO;
};

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
          index_t kBlockSize,
          index_t kHeadDim,
          index_t kM0PerBlock,
          index_t kN0PerBlock,
          index_t kK0PerBlock,
          index_t kN1PerBlock,
          index_t kK1PerBlock>
struct FlashAttentionFwd
{
    __device__ void operator()(const QDataType* q_ptr,
                               const KDataType* k_ptr,
                               const VDataType* v_ptr,
                               ODataType* o_ptr,
                               const index_t M0,
                               const index_t N0,
                               const index_t K0,
                               const index_t N1,
                               const index_t /* Batch */,
                               const index_t StrideQ,
                               const index_t StrideK,
                               const index_t StrideV,
                               const index_t StrideO,
                               const index_t BatchStrideQ,
                               const index_t BatchStrideK,
                               const index_t BatchStrideV,
                               const index_t BatchStrideO) const
    {
        const index_t id_block = get_block_id();

        const index_t num_tile_m0 = integer_divide_ceil(M0, kM0PerBlock);
        const index_t num_tile_n1 = integer_divide_ceil(N1, kN1PerBlock);

        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;
 
            return make_tuple(quotient, modulus);
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

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename ODataType>
float flash_attention_fwd(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a,
                          const stream_config& stream_config);

} // namespace ck_tile
