// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_tile.n (100) not divisible by block_waves.n * wave_tile.n (2*16=32).
// Expected error: "block_tile.n must be divisible by (block_waves.n * wave_tile.n)"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 100, 32}, {2, 2, 1}, {16, 16, 16}},
    TargetSet::cdna());
