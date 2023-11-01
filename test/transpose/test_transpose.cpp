// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_transpose_util.hpp"

using F16 = ck::half_t;
using F32 = float;

template <typename Tuple>
class TestTranspose : public ::testing::Test
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<      F16,       F16>,
    std::tuple<      F32,       F32>
    >;
// clang-format on

TYPED_TEST_SUITE(TestTranspose, KernelTypes);

//#include "test_transpose_ut_cases.inc"
