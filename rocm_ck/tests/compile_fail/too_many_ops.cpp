// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Must fail: too many operations in Signature (exceeds kMaxOps = 8)
// Expected error: "excess elements in struct initializer"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

// Create a signature with 9 operations (exceeds kMaxOps=8)
// GemmOp + 8 epilogue ops = 9 total
constexpr auto bad = resolve(Signature{.dtype = DataType::FP16,
                                       .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                 ReluOp{.in = "C", .out = "D"},
                                                 ReluOp{.in = "D", .out = "E"},
                                                 ReluOp{.in = "E", .out = "F"},
                                                 ReluOp{.in = "F", .out = "G"},
                                                 ReluOp{.in = "G", .out = "H"},
                                                 ReluOp{.in = "H", .out = "I"},
                                                 ReluOp{.in = "I", .out = "J"},
                                                 ReluOp{.in = "J", .out = "K"}}});
