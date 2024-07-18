// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include <utility>

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/utility/multi_index.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/tuple.hpp"

#include "ck/tile_program/tile/tile_gemm_shape.hpp"

namespace ck {
namespace detail {

template <typename Descriptor>
class DescToBlock2TileMapAdaptor
{
    static_assert(std::is_same_v<
                  MultiIndex<2>,
                  remove_cvref_t<decltype(std::declval<const Descriptor&>().CalculateBottomIndex(
                      std::declval<MultiIndex<1>>()))>>);

    Descriptor descriptor_;

    public:
    explicit constexpr DescToBlock2TileMapAdaptor(Descriptor descriptor)
        : descriptor_(std::move(descriptor))
    {
    }

    __host__ __device__ MultiIndex<2> operator()(index_t block_id) const
    {
        return descriptor_.CalculateBottomIndex(make_multi_index(block_id));
    }
};

template <typename Descriptor>
__host__ __device__ static auto make_desc_to_block2tile_map_adaptor(Descriptor&& descriptor)
{
    return DescToBlock2TileMapAdaptor<remove_cvref_t<Descriptor>>{
        std::forward<Descriptor>(descriptor)};
}
} // namespace detail

struct Block2TileMapNFast
{
    __host__ __device__ static constexpr auto MakeBlock2TileMap(index_t NumTilesM,
                                                                index_t NumTilesN)
    {
        return ck::detail::make_desc_to_block2tile_map_adaptor(
            make_cluster_descriptor(make_tuple(NumTilesM, NumTilesN)));
    }
};

struct Block2TileMapMFast
{
    __host__ __device__ constexpr auto operator()(index_t NumTilesM, index_t NumTilesN) const
    {
        const auto unmerge = make_merge_transform(make_tuple(NumTilesN, NumTilesM));

        return [unmerge](index_t block_id) {
            MultiIndex<2> unmerged;
            unmerge.CalculateLowerIndex(unmerged, make_multi_index(block_id));

            return make_multi_index(unmerged.At<1>(), unmerged.At<0>());
        };
    }
};

/// NOTICE: This map will be compiled into considerable amount of instructions.
///         Use with caution or replace it with more efficient implementation.
template <index_t MaxCols = 8>
struct Block2TileMapNAdapt
{
    __host__ __device__ static constexpr auto MakeBlock2TileMap(index_t NumTilesM,
                                                                index_t NumTilesN)
    {
        return [=](index_t block_id) {
            index_t idx_M0 = block_id % NumTilesM;
            index_t idx_N0 = block_id / NumTilesM;

            const auto LastCols =
                (idx_N0 < NumTilesN - NumTilesN % MaxCols) ? MaxCols : NumTilesN % MaxCols;

            index_t idx_N00          = idx_N0 / MaxCols;
            index_t idx_N01          = idx_N0 % MaxCols;
            index_t idx_M0_N01_local = idx_M0 + idx_N01 * NumTilesM;

            return make_multi_index(idx_M0_N01_local / LastCols,
                                    idx_M0_N01_local % LastCols + idx_N00 * MaxCols);
        };
    }
};

/// NOTICE: This map will be compiled into considerable amount of instructions.
///         Use with caution or replace it with more efficient implementation.
template <index_t MaxRows = 8>
struct Block2TileMapMAdapt
{
    __host__ __device__ static constexpr auto MakeBlock2TileMap(index_t NumTilesM,
                                                                index_t NumTilesN)
    {
        return [=](index_t block_id) {
            index_t idx_N0 = block_id % NumTilesN;
            index_t idx_M0 = block_id / NumTilesN;

            const auto LastRows =
                (idx_M0 < NumTilesM - NumTilesM % MaxRows) ? MaxRows : NumTilesM % MaxRows;

            index_t idx_M00          = idx_M0 / MaxRows;
            index_t idx_M01          = idx_M0 % MaxRows;
            index_t idx_N0_M01_local = idx_N0 + idx_M01 * NumTilesN;

            return make_multi_index(idx_N0_M01_local % LastRows + idx_M00 * MaxRows,
                                    idx_N0_M01_local / LastRows);
        };
    }
};

using DefaultBlock2TileMap = Block2TileMapMFast;

} // namespace ck
