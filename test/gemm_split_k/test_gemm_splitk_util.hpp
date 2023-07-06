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
#include "profiler/profile_gemm_splitk_impl.hpp"

namespace ck {
namespace test {

template <typename Tuple>
class TestGemmSplitK : public testing::Test
{
    using Row = ck::tensor_layout::gemm::RowMajor;
    using F32 = float;

    protected:
    using ALayout   = std::tuple_element_t<0, Tuple>;
    using BLayout   = std::tuple_element_t<1, Tuple>;
    using CLayout   = Row;
    using ADataType = std::tuple_element_t<2, Tuple>;
    using BDataType = std::tuple_element_t<3, Tuple>;
    using CDataType = std::tuple_element_t<4, Tuple>;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // decimal value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance
    std::vector<int> k_batches_;

    void SetUp() override { k_batches_ = {1, 2, 3, 5, 8}; }

    void Run(const int M,
             const int N,
             const int K,
             const int StrideA,
             const int StrideB,
             const int StrideC)
    {
        for(auto kb : k_batches_)
        {
            RunSingle(M, N, K, StrideA, StrideB, StrideC, kb);
        }
    }

    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   int kbatch = 1)
    {
        bool pass = ck::profiler::profile_gemm_splitk_impl<ADataType,
                                                           BDataType,
                                                           F32,
                                                           CDataType,
                                                           ALayout,
                                                           BLayout,
                                                           CLayout>(
            verify_, init_method_, log_, bench_, M, N, K, StrideA, StrideB, StrideC, kbatch);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck
