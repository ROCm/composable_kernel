// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_contraction_multi_abd_util.hpp"

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;

// clang-format off
using KernelTypes = ::testing::Types<
    //          AsDataType,                   BsDataType,                   DsDataType,                   AccDataType, EDataType, AElementWiseFn, BElementWiseFn, CDEElementWiseFn, UseCshuffleEpilog
    std::tuple<    ck_tile::tuple<F16, F16>,     ck_tile::tuple<F16>,          ck_tile::tuple<F16>,          F32,        F16,       AddScale,       PassThrough,    AddDs,            std::false_type>,
    std::tuple<    ck_tile::tuple<F16, F16>,     ck_tile::tuple<F16>,          ck_tile::tuple<F32>,          F32,        F32,       AddScale,       PassThrough,    AddDs,            std::false_type>,
    std::tuple<    ck_tile::tuple<F16, F16>,     ck_tile::tuple<F16, F16>,     ck_tile::tuple<F16>,          F32,        F16,       AddScale,       AddScale,       AddDs,            std::false_type>,
    std::tuple<    ck_tile::tuple<F16, F16>,     ck_tile::tuple<F16>,          ck_tile::tuple<F16, F16>,     F32,        F16,       AddScale,       PassThrough,    AddDs,            std::false_type>,
    std::tuple<    ck_tile::tuple<BF16, BF16>,   ck_tile::tuple<BF16>,         ck_tile::tuple<BF16>,         F32,        BF16,      AddScale,       PassThrough,    AddDs,            std::false_type>
     >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileContractionMultiABD, KernelTypes);

#include "test_contraction_multi_abd_ut_cases_default2d.inc"
