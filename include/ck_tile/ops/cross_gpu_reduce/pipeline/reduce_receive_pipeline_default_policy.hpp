// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct ReduceReceivePipelineDefaultPolicy
{
    template <typename ReduceShape>
    CK_TILE_DEVICE static constexpr auto MakeDramTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<ReduceShape::WarpPerBlock_M,
                                                      ReduceShape::MThreadPerWarp,
                                                      ReduceShape::ThreadTile_M>,
                                             sequence<ReduceShape::WarpPerBlock_N,
                                                      ReduceShape::NThreadPerWarp,
                                                      ReduceShape::ThreadTile_N>>,
                                       tuple<sequence<1, 2>, sequence<1, 2>>,
                                       tuple<sequence<0, 0>, sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<2, 2>>{});
    }

    template <typename ReduceShape>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        using namespace ck_tile;

        constexpr index_t kMPerBlock = ReduceShape::Block_M;
        constexpr index_t kNPerBlock = ReduceShape::Block_N;

        constexpr auto lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kMPerBlock, kNPerBlock), number<32>{});

        return lds_block_desc;
    }

    template <typename DataType, typename ReduceShape>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_host =
            sizeof(DataType) * MakeLdsBlockDescriptor<ReduceShape>().get_element_space_size();
        return smem_size_host * 2;
    }
};
} // namespace ck_tile
