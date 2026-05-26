// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: INT8 input with FP32 accumulator (default).
// Expected error: "INT8 GEMM requires I32 accumulator"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad =
    makeSpec(Signature{.dtype = DataType::I8, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
             GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 32}},
             TargetSet::family_gfx94());
