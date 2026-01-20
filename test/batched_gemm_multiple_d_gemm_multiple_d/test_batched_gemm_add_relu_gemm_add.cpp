// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_batched_gemm_multiple_d_gemm_multiple_d.hpp"

template <typename Tuple>
class TestBatchedGemmMultipleDGemmMultipleD
    : public BaseTestBatchedGemmMultipleDGemmMultipleD<Tuple>
{
};

using A0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDE0ElementOp = ck::tensor_operation::element_wise::AddRelu;

using B1ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDE1ElementOp = ck::tensor_operation::element_wise::Add;

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, ck::Tuple<F16>, F16, ck::Tuple<F16>, F16, Row, Col, ck::Tuple<Row>, Row, ck::Tuple<Row>, Row, A0ElementOp, B0ElementOp, CDE0ElementOp, B1ElementOp, CDE1ElementOp>,
    std::tuple<F16, F16, ck::Tuple<F16>, F16, ck::Tuple<F16>, F16, Row, Col, ck::Tuple<Row>, Col, ck::Tuple<Row>, Row, A0ElementOp, B0ElementOp, CDE0ElementOp, B1ElementOp, CDE1ElementOp>
    >;
// clang-format on

TYPED_TEST_SUITE(TestBatchedGemmMultipleDGemmMultipleD, KernelTypes);
#include "test_batched_gemm_multiple_d_gemm_multiple_d_ut_cases.inc"
