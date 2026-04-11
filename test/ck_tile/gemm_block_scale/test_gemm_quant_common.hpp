// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Common includes for all gemm quant tests
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <gtest/gtest.h>
#include <memory>

#include "test_gemm_quant_fixtures.hpp"

// Common layout aliases
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

// Common data type aliases
using Half   = ck_tile::half_t;
using FP16   = ck_tile::fp16_t;
using BF16   = ck_tile::bf16_t;
using FP8    = ck_tile::fp8_t;
using BF8    = ck_tile::bf8_t;
using E8M0   = ck_tile::e8m0_t;
using PkInt4 = ck_tile::pk_int4_t;
using PkFP4  = ck_tile::pk_fp4_t;

// Common quant type aliases
using AQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::AQuantGrouped>;
using BQuantGrouped = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::BQuantGrouped>;
using ABQuantGrouped =
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::ABQuantGrouped>;
using RowColQuant = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::RowColQuant>;
using TensorQuant = std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::TensorQuant>;

// Common group size aliases
using GroupSize1D_128 = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using GroupSize1D_64  = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;
using GroupSize2D     = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
