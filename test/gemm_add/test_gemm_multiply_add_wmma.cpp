// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "test_gemm_common.hpp"
#include "profiler/profile_gemm_multiply_add_impl.hpp"

template <typename Tuple>
class TestGemmMultiplyAdd : public TestGemmD0D1Common<Tuple>
{
    using ProfileCall = typename TestGemmD0D1Common<Tuple>::ProfileCall;

    ProfileCall GetImpl() override
    {
        return ck::profiler::profile_gemm_multiply_add_impl<
            typename TestGemmD0D1Common<Tuple>::ADataType,
            typename TestGemmD0D1Common<Tuple>::BDataType,
            typename TestGemmD0D1Common<Tuple>::AccDataType,
            typename TestGemmD0D1Common<Tuple>::D0DataType,
            typename TestGemmD0D1Common<Tuple>::D1DataType,
            typename TestGemmD0D1Common<Tuple>::EDataType,
            typename TestGemmD0D1Common<Tuple>::ALayout,
            typename TestGemmD0D1Common<Tuple>::BLayout,
            typename TestGemmD0D1Common<Tuple>::D0Layout,
            typename TestGemmD0D1Common<Tuple>::D1Layout,
            typename TestGemmD0D1Common<Tuple>::ELayout>;
    }
};

using KernelTypes =
    ::testing::Types<std::tuple<F16, F16, F32, F16, F16, F16, Row, Col, Row, Row, Row>,
                     std::tuple<F16, F16, F32, F16, F16, F16, Row, Row, Row, Row, Row>>;

TYPED_TEST_SUITE(TestGemmMultiplyAdd, KernelTypes);
TYPED_TEST(TestGemmMultiplyAdd, Test_BF16FP16) { this->Run(); }
