// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_tile.k (30) not divisible by block_waves.k * wave_tile.k (1*16=16).
// Expected error: "block_tile.k must be divisible by (block_waves.k * wave_tile.k)"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 30}, {2, 2, 1}, {16, 16, 16}},
    TargetSet::cdna());
