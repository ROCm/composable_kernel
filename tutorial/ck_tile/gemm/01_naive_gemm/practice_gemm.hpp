// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

#include "block_level/block_gemm_pipeline_agmem_bgmem_creg.hpp"
#include "host_level/grid_gemm.hpp"

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

template <index_t kMPerTile, index_t kNPerTile, index_t kKPerTile>
struct TileGemmShape
{
    static constexpr index_t kM = kMPerTile;
    static constexpr index_t kN = kNPerTile;
    static constexpr index_t kK = kKPerTile;
};

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
    static constexpr index_t kBlockSize = kBlockSize_;

    using GridGemmProblem_ =
        GridGemmProblem<ADataType, BDataType, AccDataType, CDataType, CElementFunction>;

    struct GridGemmPolicy
    {
        static constexpr index_t kBlockSize = kBlockSize_;
        static constexpr index_t kMPerBlock = kMPerBlock_;
        static constexpr index_t kNPerBlock = kNPerBlock_;
        static constexpr index_t kKPerBlock = kKPerBlock_;

        template <typename Problem>
        CK_TILE_HOST_DEVICE static constexpr auto MakeBlock2TileMap(index_t M0, index_t N0)
        {
            const auto unmerge = make_merge_transform(make_tuple(N0, M0));

            return [unmerge](index_t block_id) {
                multi_index<2> unmerged;
                unmerge.calculate_lower_index(unmerged, make_multi_index(block_id));

                return make_multi_index(unmerged.at(number<1>{}), unmerged.at(number<0>{}));
            };
        }

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
    };

    using GridGemm_ = GridGemm<GridGemmProblem_, GridGemmPolicy>;

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

        GridGemm_{}(a_dram, b_dram, c_dram, c_element_func);
    }
};

} // namespace ck_tile
