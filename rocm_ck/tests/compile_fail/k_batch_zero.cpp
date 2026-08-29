// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: k_batch = 0.
// Expected error: "k_batch must be positive"

#include <rocm_ck/gemm_spec.hpp>
using rocm_ck::TargetSet;

using namespace rocm_ck;

constexpr auto k = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{.block_tile  = {128, 128, 32},
                  .block_waves = {2, 2, 1},
                  .wave_tile   = {16, 16, 16},
                  .k_batch     = 0},
    TargetSet::cdna());
