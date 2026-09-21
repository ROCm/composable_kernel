// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: first op is not GemmOp.
// Expected error: "GEMM makeSpec requires GemmOp as first operator"

#include <rocm_ck/gemm_spec.hpp>
using rocm_ck::TargetSet;

using namespace rocm_ck;

constexpr auto bad =
    makeSpec(Signature{.dtype = DataType::FP16, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
             GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
             TargetSet::cdna());
