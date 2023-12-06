// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <numeric>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"

#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"

// Compare data in tensor with offset with layout.
// Data and offset should match if physical memory has been initialized with
// sequentially increasing values.
template <typename TensorType, typename LayoutType, typename Idxs>
__host__ __device__ bool TestTensorReadWriteCheckCustom(const TensorType& tensor,
                                                        const LayoutType& layout,
                                                        const std::vector<Idxs> idxs)
{
    for(size_t i = 0; i < idxs.size(); i++)
    {
        if(tensor(idxs[i]) != layout(idxs[i]))
        {
            return false;
        }
    }
    return true;
}

template <typename TensorType, typename LayoutType>
__host__ __device__ bool TestTensorReadWriteCheck1d(const TensorType& tensor,
                                                    const LayoutType& layout)
{
    for(ck::index_t w = 0; w < ck::wrapper::size<0>(layout); w++)
    {
        if(tensor(w) != layout(ck::make_tuple(w)))
        {
            return false;
        }
    }
    return true;
}

TEST(TestTensorReadWrite, HostMemory)
{
    constexpr ck::index_t nelems = 8;

    std::array<ck::index_t, nelems> data;
    std::iota(data.begin(), data.end(), 0);
    const auto layout = ck::wrapper::make_layout(ck::make_tuple(4, 2));
    const auto tensor = ck::wrapper::make_tensor(&data[0], layout);

    std::vector<ck::Tuple<ck::index_t, ck::index_t>> idxs;
    for(ck::index_t h = 0; h < ck::wrapper::size<0>(layout); h++)
    {
        for(ck::index_t w = 0; w < ck::wrapper::size<1>(layout); w++)
        {
            idxs.emplace_back(h, w);
        }
    }

    EXPECT_TRUE(TestTensorReadWriteCheck1d(tensor, layout));
    EXPECT_TRUE(TestTensorReadWriteCheckCustom(tensor, layout, idxs));
}

TEST(TestTensorReadWrite, HostMemoryNested)
{
    constexpr ck::index_t nelems = 8;

    std::array<ck::index_t, nelems> data;
    std::iota(data.begin(), data.end(), 0);
    const auto layout = ck::wrapper::make_layout(ck::make_tuple(ck::make_tuple(2, 2), 2));
    const auto tensor = ck::wrapper::make_tensor(&data[0], layout);

    std::vector<ck::Tuple<ck::Tuple<ck::index_t, ck::index_t>, ck::index_t>> idxs;
    for(ck::index_t d = 0; d < ck::wrapper::size<0>(ck::wrapper::get<0>(layout)); d++)
    {
        for(ck::index_t h = 0; h < ck::wrapper::size<1>(ck::wrapper::get<0>(layout)); h++)
        {
            for(ck::index_t w = 0; w < ck::wrapper::size<1>(layout); w++)
            {
                idxs.emplace_back(ck::make_tuple(d, h), w);
            }
        }
    }

    EXPECT_TRUE(TestTensorReadWriteCheck1d(tensor, layout));
    EXPECT_TRUE(TestTensorReadWriteCheckCustom(tensor, layout, idxs));
}
