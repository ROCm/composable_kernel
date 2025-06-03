// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_add_multiply_impl.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using F16  = ck::half_t;
using F32  = float;

// TODO: inerit TestGemmAddMultiply from TestGemmD0Common after changes are rebased on top of multipleD feature branch.
// After that clean test...
template <typename Tuple>
class TestGemmAddMultiply : public ::testing::Test
{
private:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using D0DataType  = std::tuple_element_t<3, Tuple>;
    using D1DataType  = std::tuple_element_t<4, Tuple>;
    using EDataType   = std::tuple_element_t<5, Tuple>;
    using ALayout     = std::tuple_element_t<6, Tuple>;
    using BLayout     = std::tuple_element_t<7, Tuple>;
    using D0Layout    = std::tuple_element_t<8, Tuple>;
    using D1Layout    = std::tuple_element_t<9, Tuple>;
    using ELayout     = std::tuple_element_t<10, Tuple>;

    constexpr static auto ProfileGemmAddMultiplyImpl =
        ck::profiler::profile_gemm_add_multiply_impl<ADataType,
                                                     BDataType,
                                                     AccDataType,
                                                     D0DataType,
                                                     D1DataType,
                                                     EDataType,
                                                     ALayout,
                                                     BLayout,
                                                     D0Layout,
                                                     D1Layout,
                                                     ELayout>;

    decltype(ProfileGemmAddMultiplyImpl) GetImpl() { return ProfileGemmAddMultiplyImpl; }

protected:
    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {16, 32, 64}, {2048, 1024, 16}, {2048, 4096, 1024}};

        bool all_success = true;

        for(auto length : lengths)
        {
            int M        = length[0];
            int N        = length[1];
            int K        = length[2];
            int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB  = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideD0 = ck::is_same_v<D0Layout, Row> ? N : M;
            int StrideD1 = ck::is_same_v<D1Layout, Row> ? N : M;
            int StrideE  = ck::is_same_v<ELayout, Row> ? N : M;

            all_success =
                all_success &
                GetImpl()(true, 1, false, false, M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE);
        }

        EXPECT_TRUE(all_success);
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16, F32, F16, F16, F16, Row, Col, Row, Row, Row>>;

TYPED_TEST_SUITE(TestGemmAddMultiply, KernelTypes);
TYPED_TEST(TestGemmAddMultiply, Test_BF16FP16) { this->Run(); }
