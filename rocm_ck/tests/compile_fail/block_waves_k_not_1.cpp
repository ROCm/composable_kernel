// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_waves.k != 1 (CShuffleEpilogue constraint).
// Expected error: "CShuffle epilogue requires block_waves.k == 1 (waves_m x waves_n layout)"

#include <rocm_ck/gemm_spec.hpp>
using rocm_ck::TargetSet;

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 2}, {16, 16, 16}},
    TargetSet::cdna());
