// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_add_impl.hpp"
#include "test_gemm_common.hpp"

template <typename Tuple>
class TestGemmAdd : public TestGemmD0Common<Tuple>
{
    private:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using D0DataType  = std::tuple_element_t<3, Tuple>;
    using EDataType   = std::tuple_element_t<4, Tuple>;
    using ALayout     = std::tuple_element_t<5, Tuple>;
    using BLayout     = std::tuple_element_t<6, Tuple>;
    using D0Layout    = std::tuple_element_t<7, Tuple>;
    using ELayout     = std::tuple_element_t<8, Tuple>;

    constexpr static auto ProfileGemmAddImpl = ck::profiler::profile_gemm_add_impl<ADataType,
                                                                                   BDataType,
                                                                                   AccDataType,
                                                                                   D0DataType,
                                                                                   EDataType,
                                                                                   ALayout,
                                                                                   BLayout,
                                                                                   D0Layout,
                                                                                   ELayout>;

    decltype(ProfileGemmAddImpl) GetImpl() override { return ProfileGemmAddImpl; }
};

using KernelTypes =
    ::testing::Types<std::tuple<F16, F16, F32, F16, F16, Row, Row, ck::Tuple<Row>, Row>,
                     std::tuple<BF16, BF16, F32, BF16, BF16, Row, Row, ck::Tuple<Row>, Row>>;

TYPED_TEST_SUITE(TestGemmAdd, KernelTypes);
TYPED_TEST(TestGemmAdd, Test_BF16FP16) { this->Run(); }
