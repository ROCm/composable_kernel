// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_mx_grouped_gemm_util.hpp"

using F8        = ck_tile::fp8_t;
using BF8       = ck_tile::bf8_t;
using F16       = ck_tile::half_t;
using F32       = float;
using BF16      = ck_tile::bf16_t;
using Row       = ck_tile::tensor_layout::gemm::RowMajor;
using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
using True      = ck_tile::bool_constant<true>;
using False     = ck_tile::bool_constant<false>;
using E8M0      = ck_tile::e8m0_t;
using Intrawave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Intrawave>;
using CompTDMV1 = ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompTDMV1>;
using CompTDMV2 = ck_tile::integral_constant<MxGemmPipelineType, MxGemmPipelineType::CompTDMV2>;
template <ck_tile::index_t N>
using ScaleBS = ck_tile::integral_constant<ck_tile::index_t, N>;

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AScaleDataType, BScaleDataType, AccDataType, CDataType, Persistent, Scheduler, PipelineType, ScaleBlockSize
std::tuple<    Row,     Col,     Row,       F8,        F8,       E8M0,          E8M0,           F32,         F16,       False,   Intrawave,        CompTDMV1,  ScaleBS<32>>,   
std::tuple<    Row,     Col,     Row,       BF8,       F8,       E8M0,          E8M0,           F32,         F16,       False,   Intrawave,        CompTDMV1,  ScaleBS<32>>,
std::tuple<    Row,     Row,     Row,       BF8,       F8,       E8M0,          E8M0,           F32,         F16,       False,   Intrawave,        CompTDMV1,  ScaleBS<32>>,
std::tuple<    Col,     Row,     Row,       F8,        BF8,      E8M0,          E8M0,           F32,         F16,       False,   Intrawave,        CompTDMV1,  ScaleBS<32>>,
std::tuple<    Row,     Col,     Row,       F8,        F8,       E8M0,          E8M0,           F32,         F16,       False,   Intrawave,        CompTDMV2,  ScaleBS<32>>, 
std::tuple<    Row,     Col,     Row,       BF8,       F8,       E8M0,          E8M0,           F32,         F16,       False,   Intrawave,        CompTDMV2,  ScaleBS<32>>,
std::tuple<    Row,     Row,     Row,       BF8,       F8,       E8M0,          E8M0,           F32,         F16,       False,   Intrawave,        CompTDMV2,  ScaleBS<32>>,
std::tuple<    Col,     Row,     Row,       F8,        BF8,      E8M0,          E8M0,           F32,         F16,       False,   Intrawave,        CompTDMV2,  ScaleBS<32>>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileMxGroupedGemm, KernelTypes);

#include "test_mx_grouped_gemm_ut_cases.inc"
