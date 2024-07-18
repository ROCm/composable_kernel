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

#include "gemm_softmax_gemm_impl.hpp"

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
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock>
struct GemmSoftmaxGemm
{
    __device__ void operator()(const QDataType* q_ptr,
                               const KDataType* k_ptr,
                               const VDataType* v_ptr,
                               ODataType* o_ptr,
                               const ck::index_t M0,
                               const ck::index_t N0,
                               const ck::index_t K0,
                               const ck::index_t N1,
                               const ck::index_t StrideQ,
                               const ck::index_t StrideK,
                               const ck::index_t StrideV,
                               const ck::index_t StrideO) const
    {
        using namespace ck;

        // divide problem
        const auto num_tile_n1 = N1 / kN1PerBlock;

        const auto id_block = get_block_id();

        const auto id_tile_m = id_block / num_tile_n1;
        const auto id_tile_n = id_block - id_tile_m * num_tile_n1;

        const auto iM0 = __builtin_amdgcn_readfirstlane(id_tile_m * kM0PerBlock);
        const auto iN1 = __builtin_amdgcn_readfirstlane(id_tile_n * kN1PerBlock);

        const auto kernel_impl = GemmSoftmaxGemmImpl<QDataType,
                                                     KDataType,
                                                     VDataType,
                                                     SaccDataType,
                                                     SMPLComputeDataType,
                                                     PDataType,
                                                     OaccDataType,
                                                     ODataType,
                                                     kBlockSize,
                                                     kM0PerBlock,
                                                     kN0PerBlock,
                                                     kK0PerBlock,
                                                     kN1PerBlock>{};

        kernel_impl(q_ptr,
                    k_ptr,
                    v_ptr,
                    o_ptr,
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
