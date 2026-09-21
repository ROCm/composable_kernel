// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: wave_tile.m is negative.
// Expected error: "wave_tile dimensions must be positive"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {-16, 16, 16}},
    TargetSet::cdna());
