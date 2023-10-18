// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/utility/data_type.hpp"
#include "profiler/profile_transpose_impl.hpp"

namespace ck {
namespace test {

template <typename Tuple>
class TestTranspose : public testing::Test
{
    using F32 = float;

    protected:
    // using ALayout   = std::tuple_element_t<0, Tuple>;
    // using BLayout   = std::tuple_element_t<1, Tuple>;
    using ADataType = std::tuple_element_t<0, Tuple>;
    using BDataType = std::tuple_element_t<1, Tuple>;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // decimal value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance

    void Run(const int N, const int C, const int D, const int H, const int W)
    {
        RunSingle(N, H, C, D, W);
    }

    void RunSingle(const int N, const int C, const int D, const int H, const int W)
    {
        bool pass = ck::profiler::profile_transpose_impl<ADataType, BDataType, >(
            verify_, init_method_, log_, bench_, N, C, D, H, W);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck