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
#include "include/ck/utility/data_type.hpp"
#include "profiler/profile_grouped_gemm_impl.hpp"

namespace ck {
namespace test {

template <typename Tuple>
class TestGroupedGemm : public testing::TestWithParam<int>
{
    protected:
    using ALayout   = std::tuple_element_t<0, Tuple>;
    using BLayout   = std::tuple_element_t<1, Tuple>;
    using ELayout   = std::tuple_element_t<2, Tuple>;
    using ADataType = std::tuple_element_t<3, Tuple>;
    using BDataType = std::tuple_element_t<4, Tuple>;
    using EDataType = std::tuple_element_t<5, Tuple>;

    public:
    bool verify_     = true;
    int init_method_ = 2; // decimal value initialization
    bool log_        = false;
    bool bench_      = false; // measure kernel performance

    void SetUp() override {}

    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             const std::vector<int>& StrideAs,
             const std::vector<int>& StrideBs,
             const std::vector<int>& StrideCs,
             int kbatch = 1)
    {
        bool pass = ck::profiler::profile_grouped_gemm_impl<ADataType,
                                                            BDataType,
                                                            EDataType,
                                                            float,
                                                            ALayout,
                                                            BLayout,
                                                            ELayout>(
            verify_, init_method_, log_, bench_, Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, kbatch);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck
