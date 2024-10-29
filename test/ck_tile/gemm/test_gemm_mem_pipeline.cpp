// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_mem_pipeline_util.hpp"

using F16 = ck_tile::half_t;
using F32 = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,      F16>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,      F16>,
    std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,      F16>,
    std::tuple<    Col,     Col,     Row,       F16,       F16,         F32,      F16>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmMemPipeline, KernelTypes);

#include "test_gemm_mem_pipeline_ut_cases.inc"
