// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_tile.m (100) not divisible by block_waves.m * wave_tile.m (2*16=32).
// Expected error: "block_tile.m must be divisible by (block_waves.m * wave_tile.m)"

#include <rocm_ck/gemm_spec.hpp>
using rocm_ck::TargetSet;

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{100, 128, 32}, {2, 2, 1}, {16, 16, 16}},
    TargetSet::cdna());
