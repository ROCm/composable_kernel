// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: tensor entry has metadata but no name.
// Expected error: "Tensor entry has metadata but no name"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(Signature{.dtype   = DataType::FP16,
                                       .tensors = {Tensor{.dtype = DataType::FP32}},
                                       .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});
