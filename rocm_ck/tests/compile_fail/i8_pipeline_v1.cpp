// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: INT8 GEMM with Pipeline::V1.
// Expected error: "INT8 GEMM requires V3/V4/Memory pipeline"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::I8,
              .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C", .acc_dtype = DataType::I32}}},
    GemmAlgorithm{.block_tile  = {128, 128, 32},
                  .block_waves = {2, 2, 1},
                  .wave_tile   = {16, 16, 32},
                  .pipeline    = Pipeline::V1},
    TargetSet::family_gfx94());
