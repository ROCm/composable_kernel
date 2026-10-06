// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_mx_gemm_pipeline_kernel_types.hpp"
#include "test_mx_gemm_pipeline_wmma_base.hpp"

// The bridge uses E8M0 per 32 K elements and the non-cluster 16x16 WMMA path.
// Unlike the 32x32 packed-FP4 instruction, these variants support revision 0.
using BridgeE8M0Types = ::testing::Types<std::tuple<Row,
                                                    Col,
                                                    Row,
                                                    F8,
                                                    F8,
                                                    E8M0,
                                                    E8M0,
                                                    F32,
                                                    F16,
                                                    I64,
                                                    I64,
                                                    I128,
                                                    I16,
                                                    I16,
                                                    CompTDMV1,
                                                    I32>,
                                         std::tuple<Row,
                                                    Col,
                                                    Row,
                                                    F4,
                                                    F4,
                                                    E8M0,
                                                    E8M0,
                                                    F32,
                                                    F16,
                                                    I64,
                                                    I64,
                                                    I128,
                                                    I16,
                                                    I16,
                                                    CompTDMV1,
                                                    I32>>;

template <typename T>
class TestCkTileMxGemmE8M0Gfx1250
    : public TestCkTileMxGemmPipelineWmmaBase<T, TestCkTileMxGemmE8M0Gfx1250<T>>
{
};

TYPED_TEST_SUITE(TestCkTileMxGemmE8M0Gfx1250, BridgeE8M0Types);

#define TEST_SUITE_NAME TestCkTileMxGemmE8M0Gfx1250
#include "test_mx_gemm_pipeline_ut_cases.inc"
#undef TEST_SUITE_NAME
