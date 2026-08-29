// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Must fail: attempts to create more epilogue AddOps than supported
// Expected error: "maximum 2 D tensors in epilogue chain"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

// Try to create 3 consecutive AddOps after GemmOp
// GEMM epilogue allows: up to 2 AddOps (for D0, D1), then 1 unary op
// Third AddOp will trigger "unexpected operator after GEMM epilogue chain"
constexpr auto bad = makeSpec(
    Signature{.dtype   = DataType::FP16,
              .tensors = {Tensor{.name = "D0"}, Tensor{.name = "D1"}, Tensor{.name = "D2"}},
              .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                          AddOp{.lhs = "C", .rhs = "D0", .out = "E0"},
                          AddOp{.lhs = "E0", .rhs = "D1", .out = "E1"},
                          AddOp{.lhs = "E1", .rhs = "D2", .out = "E2"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
    TargetSet::cdna());
