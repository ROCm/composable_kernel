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
    using ADataType = std::tuple_element_t<0, Tuple>;
    using BDataType = std::tuple_element_t<1, Tuple>;

    public:
    static constexpr bool verify_              = true;
    static constexpr int init_method_          = 1; // decimal value initialization
    static constexpr bool log_                 = false;
    static constexpr bool bench_               = false; // measure kernel performance
    std::vector<std::vector<index_t>> lengths_ = {{16, 32, 16, 32, 16}, {16, 8, 16, 32, 8}};

    void Run()
    {
        for(auto length : this->lengths_)
        {
            this->RunSingle(length);
        }
    }

    void RunSingle()
    {
        bool pass = ck::profiler::profile_transpose_impl<ADataType, BDataType, 5>(
            verify_, init_method_, log_, bench_, lengths_);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck
