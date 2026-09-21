// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "test_grouped_gemm_util.hpp"

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F8   = ck::f8_t;
using I8   = int8_t;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::FastGelu;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
class TestGroupedGemm : public ck::test::TestGroupedGemm<Tuple, true>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    ck::Tuple<     Row, Row, Row, F16, F16, F16, AElementOp, BElementOp, CDEElementOp>,
    ck::Tuple<     Row, Col, Row, F16, F16, F16, AElementOp, BElementOp, CDEElementOp>,
    ck::Tuple<     Col, Row, Row, F16, F16, F16, AElementOp, BElementOp, CDEElementOp>,
    ck::Tuple<     Col, Col, Row, F16, F16, F16, AElementOp, BElementOp, CDEElementOp>
>;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemm, KernelTypes);

#include "test_grouped_gemm_ut_cases.inc"
