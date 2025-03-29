// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "block_gemm_pipeline_agmem_bgmem_creg.hpp"
#include "config.h"
#include "grid_gemm.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename CElementFunction_>
struct GridGemmProblem
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using AccDataType = AccDataType_;
    using CDataType   = CDataType_;

    using CElementFunction = CElementFunction_;
};

#ifndef ENABLE_INSTRUCTION_SCH
template <index_t kMPerTile, index_t kNPerTile, index_t kKPerTile>
struct TileGemmShape
{
    static constexpr index_t kM = kMPerTile;
    static constexpr index_t kN = kNPerTile;
    static constexpr index_t kK = kKPerTile;
};
#endif

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          index_t kBlockSize_,
          typename BlockGemmShape_>
struct BlockGemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kBlockSize = kBlockSize_;
};

// C = A * B
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename CElementFunction,
          index_t kAAlignment,
          index_t kBAlignment,
          index_t kCAlignment,
          index_t kBlockSize_,
          index_t kMPerBlock_,
          index_t kNPerBlock_,
          index_t kKPerBlock_>
struct Gemm
{
    using GridGemmProblem = GridGemmProblem<ADataType,
                                            BDataType,
                                            AccDataType,
                                            CDataType,
                                            CElementFunction>;

    struct GridGemmPolicy
    {
        static constexpr index_t kBlockSize = kBlockSize_;
        static constexpr index_t kMPerBlock = kMPerBlock_;
        static constexpr index_t kNPerBlock = kNPerBlock_;
        static constexpr index_t kKPerBlock = kKPerBlock_;

        template <typename Problem>
        CK_TILE_HOST_DEVICE static constexpr auto MakeBlock2TileMap(index_t M0,
                                                                    index_t N0)
        {
#if defined(ENABLE_CACHE_AWARE_WG_SCH)
#pragma message ("Cache-aware work group sch")
            return [=](index_t block_1d_id) {
                constexpr index_t M01 = 4;
                constexpr index_t GroupNum = 8;

                const auto update_N0 = ((((N0 / 2) * 2) / 2) / M01) * M01 * 2;
                const auto update_M0 = ((M0 / (GroupNum / 2)) * (GroupNum / 2)) / GroupNum / M01 * M01 * GroupNum;

                const auto xcd_id = block_1d_id % GroupNum;

                const auto l_block_id = block_1d_id - (xcd_id % 2);

                const auto ridn = GroupNum * M01 * (update_N0 / 2);
                const auto rid = (l_block_id - (l_block_id % GroupNum)) / ridn;
                const auto lu = (l_block_id % GroupNum) + rid * ridn;

                const auto sub_N0_id  = (l_block_id - lu) / (GroupNum * M01);
                const auto sub_M0_id = (l_block_id - (sub_N0_id * (GroupNum * M01)+ lu ) ) / GroupNum;

                auto n = sub_N0_id + (xcd_id % 2) * (update_N0 / 2);
                auto m = rid * M01 + sub_M0_id + (update_M0 / (GroupNum / 2)) * (xcd_id / 2);

                const auto total_update_size = update_N0 * update_M0;

                if (block_1d_id >= total_update_size) {
                    auto x = (block_1d_id + 1) - total_update_size;
                    auto rlen = N0 - update_N0;

                    auto rm = 0;
                    auto rn = 0;
                    if (rlen > 0) {
                        rm = (x - 1) / rlen;
                        rn = x % rlen;
                    }

                    if (rlen > 0 and rm < M0) {
                        n = rn + update_N0;
                        m = rm;
                    } else {
                        x = x - rlen * M0;
                        rm = (x - 1) / update_N0;
                        rn = x % update_N0;
                        n = rn;
                        m = update_M0 + rm;
                    }
                }
                return make_multi_index(m, n);
	    };
#else
            const auto unmerge = make_merge_transform(make_tuple(N0, M0));

            return [unmerge](index_t block_id) {
                multi_index<2> unmerged;
                unmerge.calculate_lower_index(unmerged, make_multi_index(block_id));

                return make_multi_index(unmerged.at(number<1>{}), unmerged.at(number<0>{}));

            };
#endif
        }

#ifndef ENABLE_INSTRUCTION_SCH
        template <typename Problem>
        CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemmPipeline()
        {
            using BlockGemmPipelineProblem_ =
                BlockGemmPipelineProblem<ADataType,
                                         BDataType,
                                         AccDataType,
                                         kBlockSize,
                                         TileGemmShape<kMPerBlock, kNPerBlock, kKPerBlock>>;

            return BlockGemmPipelineAGmemBGmemCReg<BlockGemmPipelineProblem_>{};
        }
#endif
    };

    using GridGemm = GridGemm<GridGemmProblem, GridGemmPolicy>;

    CK_TILE_DEVICE void operator()(const ADataType* p_a,
                               const BDataType* p_b,
                               CDataType* p_c,
                               const index_t M,
                               const index_t N,
                               const index_t K,
                               const index_t Lda,
                               const index_t Ldb,
                               const index_t Ldc,
                               const CElementFunction& c_element_func) const
    {
        const auto a_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_a, make_tuple(M, K), make_tuple(Lda, 1), number<kAAlignment>{}, number<1>{});
        }();

        const auto b_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_b, make_tuple(N, K), make_tuple(Ldb, 1), number<kBAlignment>{}, number<1>{});
        }();

        const auto c_dram = [&] {
            return make_naive_tensor_view<address_space_enum::global>(
                p_c, make_tuple(M, N), make_tuple(Ldc, 1), number<kCAlignment>{}, number<1>{});
        }();

        GridGemm{}(a_dram, b_dram, c_dram, c_element_func);
    }
};

} // namespace ck_tile
