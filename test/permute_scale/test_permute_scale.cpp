// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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

    constexpr bool skip_case()
    {
#ifndef CK_ENABLE_FP16
        if constexpr(ck::is_same_v<ADataType, F16> || ck::is_same_v<BDataType, F16>)
        {
            return true;
        }
#endif
#ifndef CK_ENABLE_FP32
        if constexpr(ck::is_same_v<ADataType, F32> || ck::is_same_v<BDataType, F32>)
        {
            return true;
        }
#endif
        return false;
    }

    template <ck::index_t NDims>
    void Run(std::vector<ck::index_t> lengths,
             std::vector<ck::index_t> input_strides,
             std::vector<ck::index_t> output_strides)
    {
        if(!skip_case())
        {
            bool success = ck::profiler::profile_permute_scale_impl<ADataType, BDataType, NDims>(
                true, 2, false, false, lengths, input_strides, output_strides);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16>, std::tuple<F32, F32>>;

TYPED_TEST_SUITE(TestPermute, KernelTypes);
TYPED_TEST(TestPermute, Test1D)
{
    constexpr ck::index_t NumDims = 1;
    this->template Run<NumDims>({8}, {1}, {2});
    this->template Run<NumDims>({8}, {2}, {1});
    this->template Run<NumDims>({1}, {1}, {1});
}

TYPED_TEST(TestPermute, Test2D)
{
    constexpr ck::index_t NumDims = 2;
    this->template Run<NumDims>({8, 4}, {4, 1}, {1, 8});
    this->template Run<NumDims>({8, 4}, {1, 8}, {4, 1});
    this->template Run<NumDims>({1, 1}, {1, 1}, {1, 1});
}

TYPED_TEST(TestPermute, Test3D)
{
    constexpr ck::index_t NumDims = 3;
    this->template Run<NumDims>({2, 4, 4}, {16, 4, 1}, {1, 2, 8});
    this->template Run<NumDims>({2, 4, 4}, {1, 2, 8}, {16, 4, 1});
    this->template Run<NumDims>({1, 1, 1}, {1, 1, 1}, {1, 1, 1});
}

TYPED_TEST(TestPermute, Test4D)
{
    constexpr ck::index_t NumDims = 4;
    this->template Run<NumDims>({2, 4, 4, 4}, {64, 16, 4, 1}, {1, 2, 8, 32});
    this->template Run<NumDims>({2, 4, 4, 4}, {1, 2, 8, 32}, {64, 16, 4, 1});
    this->template Run<NumDims>({1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
}

TYPED_TEST(TestPermute, Test5D)
{
    constexpr ck::index_t NumDims = 5;
    this->template Run<NumDims>({2, 4, 4, 4, 4}, {256, 64, 16, 4, 1}, {1, 2, 8, 32, 128});
    this->template Run<NumDims>({2, 4, 4, 4, 4}, {1, 2, 8, 32, 128}, {256, 64, 16, 4, 1});
    this->template Run<NumDims>({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1});
}

TYPED_TEST(TestPermute, Test6D)
{
    constexpr ck::index_t NumDims = 6;
    this->template Run<NumDims>(
        {2, 4, 4, 4, 4, 4}, {1024, 256, 64, 16, 4, 1}, {1, 2, 8, 32, 128, 512});
    this->template Run<NumDims>(
        {2, 4, 4, 4, 4, 4}, {1, 2, 8, 32, 128, 512}, {1024, 256, 64, 16, 4, 1});
    this->template Run<NumDims>({1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1});
}
