// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: INT8 GEMM with gfx90a in target set.
// Expected error: "INT8 GEMM requires gfx942+"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::I8,
              .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C", .acc_dtype = DataType::I32}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 32}},
    GpuTarget::gfx90a);
