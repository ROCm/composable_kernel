// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "block_gemm_pipeline_problem.hpp"
#include "block_gemm_areg_bsmem_creg_v1.hpp"
#include "flash_attention_fwd_impl.hpp"

namespace ck_tile {

CK_TILE_HOST_DEVICE static constexpr auto MakeBlock2TileMap(index_t M0, index_t N0)
{
    return [=](index_t block_1d_id) {
        constexpr index_t M01      = 4;
        constexpr index_t GroupNum = 8;

        const auto update_N0 = ((((N0 / 2) * 2) / 2) / M01) * M01 * 2;
        const auto update_M0 =
            ((M0 / (GroupNum / 2)) * (GroupNum / 2)) / GroupNum / M01 * M01 * GroupNum;

        const auto xcd_id = block_1d_id % GroupNum;

        const auto l_block_id = block_1d_id - (xcd_id % 2);

        const auto ridn = GroupNum * M01 * (update_N0 / 2);
        const auto rid  = (l_block_id - (l_block_id % GroupNum)) / ridn;
        const auto lu   = (l_block_id % GroupNum) + rid * ridn;

        const auto sub_N0_id = (l_block_id - lu) / (GroupNum * M01);
        const auto sub_M0_id = (l_block_id - (sub_N0_id * (GroupNum * M01) + lu)) / GroupNum;

        auto n = sub_N0_id + (xcd_id % 2) * (update_N0 / 2);
        auto m = rid * M01 + sub_M0_id + (update_M0 / (GroupNum / 2)) * (xcd_id / 2);

        const auto total_update_size = update_N0 * update_M0;

        if(block_1d_id >= total_update_size)
        {
            auto x    = (block_1d_id + 1) - total_update_size;
            auto rlen = N0 - update_N0;

            auto rm = 0;
            auto rn = 0;
            if(rlen > 0)
            {
                rm = (x - 1) / rlen;
                rn = x % rlen;
            }

            if(rlen > 0 and rm < M0)
            {
                n = rn + update_N0;
                m = rm;
            }
            else
            {
                x  = x - rlen * M0;
                rm = (x - 1) / update_N0;
                rn = x % update_N0;
                n  = rn;
                m  = update_M0 + rm;
            }
        }
        return make_multi_index(m, n);
    };
}

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

#if defined(TOY_FA_FWD_CACHE_AWARE)
#pragma message("Enable toy FA fwd cache aware")
        const auto block2tile = MakeBlock2TileMap(num_tile_m0, num_tile_n1);

        const index_t id_tile_batch = id_block / num_tile_n1 / num_tile_m0;
        const auto id_tile = block2tile(id_block - id_tile_batch * num_tile_n1 * num_tile_m0);

        const index_t iBatch = __builtin_amdgcn_readfirstlane(id_tile_batch);
        const index_t iM0    = __builtin_amdgcn_readfirstlane(id_tile.template get(number<0>{}) %
                                                           num_tile_m0 * kM0PerBlock);
        const index_t iN1    = __builtin_amdgcn_readfirstlane(id_tile.template get(number<1>{}) %
                                                           num_tile_n1 * kN1PerBlock);

#else
        const auto f = [](index_t dividend, index_t divisor) {
            index_t quotient = dividend / divisor;
            index_t modulus  = dividend - quotient * divisor;

            return make_tuple(quotient, modulus);
        };
        const auto [itmp, id_tile_n]          = f(id_block, num_tile_n1);
        const auto [id_tile_batch, id_tile_m] = f(itmp, num_tile_m0);
        const index_t iBatch                  = __builtin_amdgcn_readfirstlane(id_tile_batch);
        const index_t iM0 = __builtin_amdgcn_readfirstlane(id_tile_m * kM0PerBlock);
        const index_t iN1 = __builtin_amdgcn_readfirstlane(id_tile_n * kN1PerBlock);

#endif

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

} // namespace ck_tile
