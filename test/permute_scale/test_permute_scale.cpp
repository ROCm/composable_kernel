// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_permute_scale_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestPermute : public ::testing::Test
{
    protected:
    using ADataType = std::tuple_element_t<0, Tuple>;
    using BDataType = std::tuple_element_t<1, Tuple>;

    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {4, 2, 1, 8}, {1, 1, 1, 1}, {16, 8, 32, 64}, {32, 64, 128, 128}};

        for(auto length : lengths)
        {
            bool success =
                ck::test_permute_scale_impl<ADataType, BDataType, 4>(true, 2, false, false, length);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16>, std::tuple<F32, F32>>;

TYPED_TEST_SUITE(TestPermute, KernelTypes);
TYPED_TEST(TestPermute, Test_FP16) { this->Run(); }
TYPED_TEST(TestPermute, Test_FP32) { this->Run(); }
