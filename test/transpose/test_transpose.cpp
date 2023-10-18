// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_tranpose_util.hpp"

using F16 = ck::half_t;
using F32 = float;

enum struct MatrixLayout
{
    NCDHW, // 0
    NCHWD, // 1
};

template <typename Tuple>
class TestTranspose : public ck::test::TestTranspose<typename MatrixLayout<NCDHW>::type>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<      F16,       F16>,
    std::tuple<      F32,       F32>
    >;
// clang-format on

TYPED_TEST_SUITE(TestGemmSplitK_MK_KN, KernelTypes);

//#include "test_transpose_ut_cases.inc"