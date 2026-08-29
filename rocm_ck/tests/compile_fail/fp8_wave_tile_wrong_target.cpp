// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: FP8 wave tile on gfx90a (no FP8 MFMA support).
// Expected error: "wave_tile is not a valid instruction shape for this dtype and target"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::FP8_FNUZ, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 64}, {2, 2, 1}, {32, 32, 16}},
    GpuTarget::gfx90a);
