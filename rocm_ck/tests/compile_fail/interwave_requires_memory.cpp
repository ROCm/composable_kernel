// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: Interwave scheduling with V1 pipeline (requires Memory).
// Expected error: "Interwave scheduling requires Pipeline::Memory"

#include <rocm_ck/gemm_spec.hpp>
using rocm_ck::TargetSet;

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{.block_tile         = {128, 128, 32},
                  .block_waves        = {2, 2, 1},
                  .wave_tile          = {16, 16, 16},
                  .pipeline           = Pipeline::V1,
                  .pipeline_scheduler = PipelineScheduler::Interwave},
    TargetSet::cdna());
