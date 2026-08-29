// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: rhs dtype is I4 but Tensor.quantize is not set.
// Expected error: "rhs dtype is I4 but Tensor.quantize is not set"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(Signature{.dtype   = DataType::FP16,
                                        .tensors = {Tensor{.name = "B", .dtype = DataType::I4}},
                                        .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
                              GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}},
                              TargetSet::cdna());
