// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceGroupedGemm_Fixed_NK_Common
{
    template <typename UnderlyingBlockToCTileMap, bool HasSplitKSupport = true>
    struct OffsettedBlockToCTileMapMLoops
    {
        using underlying_type = UnderlyingBlockToCTileMap;

        __host__ __device__ OffsettedBlockToCTileMapMLoops(
            UnderlyingBlockToCTileMap block_to_ctile_map, index_t block_start, index_t id_off = 0)
        {
            block_to_ctile_map_ = block_to_ctile_map;
            block_start_        = block_start;
            id_off_             = id_off;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            auto idx_bot = block_to_ctile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[Number<0>{}] - block_start_ + id_off_));

            // Workarounds the fact that gridwise gemm implementations not supporting splitk require
            // different index mapping.
            if constexpr(HasSplitKSupport)
            {
                return make_tuple(idx_bot[Number<0>{}], idx_bot[Number<1>{}], idx_bot[Number<2>{}]);
            }
            else
            {
                return make_tuple(idx_bot[Number<1>{}], idx_bot[Number<2>{}]);
            }
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return block_to_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        template <typename CGridDesc_M_N>
        __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_to_ctile_map_.CheckValidity(c_grid_desc_m_n);
        }

        template <typename CGridDesc_M_N>
        __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_to_ctile_map_.CalculateGridSize(c_grid_desc_m_n);
        }

        UnderlyingBlockToCTileMap block_to_ctile_map_;
        index_t block_start_;
        index_t id_off_;
    };

    template <index_t MPerBlock, index_t NPerBlock>
    struct BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops
    {
        static constexpr auto I0 = Number<0>{};
        static constexpr auto I1 = Number<1>{};

        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops() = default;

        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            const BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&
        operator=(const BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&) = default;
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&
        operator=(BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops&&) = default;

        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(index_t M,
                                                                          index_t N,
                                                                          index_t KBatch,
                                                                          index_t M01 = 8)
            : M_(M), N_(N), KBatch_(KBatch), M01_(M01)
        {
        }

        template <typename CGridDesc_M_N>
        __host__ __device__ BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
            const CGridDesc_M_N& c_grid_desc_m_n, index_t KBatch, index_t M01 = 8)
            : BlockToCTileMap_KBatch_M00_N0_M01Adapt_MLoops(
                  c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1), KBatch, M01)
        {
        }

        __host__ __device__ constexpr index_t CalculateGridSize(index_t M, index_t N) const
        {
            const auto M0 = math::integer_divide_ceil(M, MPerBlock);
            const auto N0 = math::integer_divide_ceil(N, NPerBlock);

            return M0 * N0 * KBatch_;
        }

        template <typename CGridDesc_M_N>
        __host__ __device__ constexpr index_t
        CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return CalculateGridSize(c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1));
        }

        template <typename CGridDesc_M_N>
        __host__ bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
        {
            return true;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            auto block_1d_id = idx_top[I0];

            const auto M0 = math::integer_divide_ceil(M_, MPerBlock);
            const auto N0 = math::integer_divide_ceil(N_, NPerBlock);

            block_1d_id = block_1d_id % (M0 * N0 * KBatch_); // hide groups

            const index_t idx_ksplit = block_1d_id / (M0 * N0);
            block_1d_id              = block_1d_id % (M0 * N0);

            index_t idx_N0 = block_1d_id % N0;
            index_t idx_M0 = block_1d_id / N0;

            const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

            index_t idx_M00          = idx_M0 / M01_;
            index_t idx_M01          = idx_M0 % M01_;
            index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

            return make_tuple(idx_ksplit,
                              idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                              idx_N0_M01_local / M01_adapt);
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                                 const CTileDim& /* c_tile_dim */) const
        {
            return true; // always valid provided that user gets grid size from CalculateGridSize()
        }

        private:
        index_t M_;
        index_t N_;
        index_t KBatch_;
        index_t M01_;
    };
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
