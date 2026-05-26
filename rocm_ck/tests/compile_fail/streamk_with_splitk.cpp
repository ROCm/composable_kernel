// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: Stream-K tile partitioning is incompatible with split-K (k_batch > 1).
// Expected error: "Stream-K tile partitioning is incompatible with split-K"

#include <rocm_ck/gemm_spec.hpp>
using rocm_ck::TargetSet;

using namespace rocm_ck;

constexpr auto bad = makeSpec(
    Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{.block_tile       = {128, 128, 32},
                  .block_waves      = {2, 2, 1},
                  .wave_tile        = {16, 16, 16},
                  .k_batch          = 4,
                  .tile_partitioner = TilePartitioner::StreamK},
    TargetSet::cdna());
