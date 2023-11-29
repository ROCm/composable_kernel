// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "profiler/profile_permute_scale_impl.hpp"

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
        std::vector<std::vector<ck::index_t>> lengths = {{4, 2, 1, 8}, {4, 2, 8, 8}};

        for(auto length : lengths)
        {
            bool success = ck::profiler::profile_permute_scale_impl<ADataType, BDataType, 4>(
                true, 2, false, false, length);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16>>;

TYPED_TEST_SUITE(TestPermute, KernelTypes);
TYPED_TEST(TestPermute, Test_FP16) { this->Run(); }
