// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <numeric>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"

TEST(TestPartition, LocalPartition)
{
    const auto shape =
        ck::make_tuple(ck::make_tuple(ck::Number<16>{}, ck::Number<4>{}), ck::Number<4>{});
    const auto strides =
        ck::make_tuple(ck::make_tuple(ck::Number<1>{}, ck::Number<16>{}), ck::Number<64>{});
    const auto layout = ck::wrapper::make_layout(shape, strides);

    std::vector<ck::index_t> data(ck::wrapper::size(layout));
    std::iota(data.begin(), data.end(), 0);

    const auto tensor =
        ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Generic>(data.data(), layout);

    const auto thread_steps =
        ck::make_tuple(ck::make_tuple(ck::Number<2>{}, ck::Number<1>{}), ck::Number<1>{});
    const auto thread_layout =
        ck::make_tuple(ck::make_tuple(ck::Number<8>{}, ck::Number<1>{}), ck::Number<1>{});

    for(ck::index_t thread_id = 0; thread_id < ck::wrapper::size(thread_layout); thread_id++)
    {
        const auto raked_partition =
            ck::wrapper::make_local_partition(tensor, thread_layout, thread_id);

        const auto expected_partition_size =
            ck::wrapper::size(tensor) / ck::wrapper::size(thread_layout);
        EXPECT_EQ(ck::wrapper::size(raked_partition), expected_partition_size);
        EXPECT_EQ(raked_partition(0), thread_id);
    }

    for(ck::index_t thread_id = 0; thread_id < ck::wrapper::size(thread_layout); thread_id++)
    {
        const auto packed_partition =
            ck::wrapper::make_local_partition(tensor, thread_layout, thread_id, thread_steps);

        const auto expected_partition_size =
            ck::wrapper::size(tensor) / ck::wrapper::size(thread_layout);
        const auto expected_partition_first_val = thread_id * ck::wrapper::size<0, 0>(thread_steps);
        EXPECT_EQ(ck::wrapper::size(packed_partition), expected_partition_size);
        EXPECT_EQ(packed_partition(0), expected_partition_first_val);
    }
}

TEST(TestPartition, LocalTile)
{
    const auto shape =
        ck::make_tuple(ck::make_tuple(ck::Number<16>{}, ck::Number<4>{}), ck::Number<4>{});
    const auto strides =
        ck::make_tuple(ck::make_tuple(ck::Number<1>{}, ck::Number<16>{}), ck::Number<64>{});
    const auto layout = ck::wrapper::make_layout(shape, strides);

    std::vector<ck::index_t> data(ck::wrapper::size(layout));
    std::iota(data.begin(), data.end(), 0);

    const auto tensor =
        ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Generic>(data.data(), layout);

    const auto block_steps =
        ck::make_tuple(ck::make_tuple(ck::Number<4>{}, ck::Number<2>{}), ck::Number<2>{});
    const auto block_shape =
        ck::make_tuple(ck::make_tuple(ck::Number<4>{}, ck::Number<2>{}), ck::Number<2>{});
    const auto block_layout =
        ck::make_tuple(ck::make_tuple(ck::Number<4>{}, ck::Number<2>{}), ck::Number<2>{});

    std::vector<ck::Tuple<ck::Tuple<ck::index_t, ck::index_t>, ck::index_t>> block_idxs;
    for(ck::index_t x = 0; x < ck::wrapper::size<0, 0>(block_layout); x++)
    {
        for(ck::index_t y = 0; y < ck::wrapper::size<0, 1>(block_layout); y++)
        {
            for(ck::index_t z = 0; z < ck::wrapper::size<1>(block_layout); z++)
            {
                block_idxs.emplace_back(ck::make_tuple(x, y), z);
            }
        }
    }

    for(const auto& block_idx : block_idxs)
    {
        const auto raked_tile = ck::wrapper::make_local_tile(tensor, block_shape, block_idx);

        const auto expected_tile_size = ck::wrapper::size(block_shape);
        EXPECT_EQ(ck::wrapper::size(raked_tile), expected_tile_size);
        EXPECT_EQ(raked_tile(0), layout(block_idx));
    }

    for(const auto& block_idx : block_idxs)
    {
        const auto packed_tile =
            ck::wrapper::make_local_tile(tensor, block_shape, block_idx, block_steps);

        const auto expected_tile_size = ck::wrapper::size(block_shape);
        const auto expected_tile_first_val =
            ck::wrapper::size<0, 0>(block_idx) * ck::wrapper::size<0, 0>(block_shape) *
                ck::wrapper::size<0, 0>(strides) +
            ck::wrapper::size<0, 1>(block_idx) * ck::wrapper::size<0, 1>(block_shape) *
                ck::wrapper::size<0, 1>(strides) +
            ck::wrapper::size<1>(block_idx) * ck::wrapper::size<1>(block_shape) *
                ck::wrapper::size<1>(strides);
        EXPECT_EQ(ck::wrapper::size(packed_tile), expected_tile_size);
        EXPECT_EQ(packed_tile(0), expected_tile_first_val);
    }
}
