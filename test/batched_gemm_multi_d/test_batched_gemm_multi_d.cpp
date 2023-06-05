// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <gtest/gtest.h>

#include "profiler/profile_batched_gemm_impl.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm_multi_d.hpp"

namespace {
using F16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Empty_Tuple = ck::Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
} // namespace

template <typename Tuple>
class TestBatchedGemmMultiD : public ::testing::Test
{
    protected:
    using ALayout  = std::tuple_element_t<0, Tuple>;
    using BLayout  = std::tuple_element_t<1, Tuple>;
    using CLayout  = std::tuple_element_t<2, Tuple>;
    using DataType = std::tuple_element_t<3, Tuple>;

    static constexpr int M          = 512;
    static constexpr int N          = 256;
    static constexpr int K          = 128;
    static constexpr int BatchCount = 3;

    void Run()
    {
        using namespace ck::tensor_operation::device;

        const bool pass =
            ck::profiler::profile_batched_gemm_impl<DataType,
                                                    DataType,
                                                    DataType,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough,
                                                    DeviceBatchedGemmMultiD<ALayout,
                                                                            BLayout,
                                                                            Empty_Tuple,
                                                                            CLayout,
                                                                            DataType,
                                                                            DataType,
                                                                            Empty_Tuple,
                                                                            DataType,
                                                                            PassThrough,
                                                                            PassThrough,
                                                                            PassThrough>>(
                true, 1, false, 1, M, N, K, K, N, N, M * K, K * N, M * N, BatchCount);
        EXPECT_TRUE(pass);
    }
};

template <typename Tuple>
class TestBatchedGemmMultiDF16 : public TestBatchedGemmMultiD<Tuple>
{
};

template <typename Tuple>
class TestBatchedGemmMultiDI8 : public TestBatchedGemmMultiD<Tuple>
{
};

using F16KernelTypes = ::testing::Types<std::tuple<Row, Row, Row, F16>,
                                        std::tuple<Row, Col, Row, F16>,
                                        std::tuple<Col, Row, Row, F16>,
                                        std::tuple<Col, Col, Row, F16>>;

using I8KernelTypes = ::testing::Types<std::tuple<Row, Row, Row, int8_t>,
                                       std::tuple<Row, Col, Row, int8_t>,
                                       std::tuple<Col, Row, Row, int8_t>,
                                       std::tuple<Col, Col, Row, int8_t>>;

TYPED_TEST_SUITE(TestBatchedGemmMultiDF16, F16KernelTypes);
TYPED_TEST_SUITE(TestBatchedGemmMultiDI8, I8KernelTypes);

TYPED_TEST(TestBatchedGemmMultiDF16, bilinear) { this->Run(); }

TYPED_TEST(TestBatchedGemmMultiDI8, scale) { this->Run(); }
