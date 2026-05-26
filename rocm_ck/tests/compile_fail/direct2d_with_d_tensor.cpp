// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: Direct2D store strategy with AddOp (D tensor).
// Expected error: "Direct2D epilogue does not support D tensors"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(Signature{.dtype = DataType::FP16,
                                        .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                  AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                              GemmAlgorithm{.block_tile     = {128, 128, 32},
                                            .block_waves    = {2, 2, 1},
                                            .wave_tile      = {16, 16, 16},
                                            .store_strategy = StoreStrategy::Direct2D},
                              TargetSet::cdna());
