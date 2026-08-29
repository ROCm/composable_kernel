// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Must fail: too many epilogue operations (exceeds kMaxEpilogueOps = 4)
// Expected error: "too many epilogue operations (max 4)"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

// Create a GEMM with 5 epilogue operations (exceeds kMaxEpilogueOps=4)
// GemmOp (idx 0) + AddOp + 4 unary ops = 6 ops total, 5 epilogue ops
constexpr auto bad = makeSpec(Signature{.dtype   = DataType::FP16,
                                        .tensors = {Tensor{.name = "D"}},
                                        .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "D", .out = "E"},
                                                    ReluOp{.in = "E", .out = "F"},
                                                    ReluOp{.in = "F", .out = "G"},
                                                    ReluOp{.in = "G", .out = "H"},
                                                    ReluOp{.in = "H", .out = "I"}}},
                              GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                              TargetSet::cdna());
