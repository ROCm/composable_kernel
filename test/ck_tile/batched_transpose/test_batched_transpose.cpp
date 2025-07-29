// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

template<typename DataType>
class TestCkTileBatchedTranspose : public ::testing::TestWithParam<std::tuple<int, int, int, int>>
{
    protected:
    void Run(std::tuple<int, int, int, int> param)
    {
        auto [N, H, W, C] = param;
        ck_tile::HostTensor<DataType> x_host(
        {N, H, W, C},
        {H * W * C, W * C, C, 1});
        ck_tile::HostTensor<DataType> y_host(
            {N, C, H, W},
            {C * H * W, H * W, W, 1});

        ck_tile::FillUniformDistribution<DataType>{-.5f, .5f}(x_host);

        ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem y_dev(y_host.get_element_space_size_in_bytes());

        bool pass = ck_tile::check_err(y_host, y_host);

        EXPECT_TRUE(pass);
    }
};

class TestCkTileBatchedTransposeHalf : public TestCkTileBatchedTranspose<ck_tile::half_t>
{
};

TEST_P(TestCkTileBatchedTransposeHalf, TestCorrectness)
{
    auto [N, H, W, C] = GetParam();
    this->Run({N, H, W, C});
}

INSTANTIATE_TEST_SUITE_P(TestCkTileBatchedTransposeSuite,
                         TestCkTileBatchedTransposeHalf,
                         ::testing::Values(std::tuple{1, 64, 1, 64}));
