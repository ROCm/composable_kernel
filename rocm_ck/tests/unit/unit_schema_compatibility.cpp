// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Schema compatibility tests: frozen baseline configs from example 04.
//
// These tests verify that schema changes (new fields, modified defaults,
// validation rules) do NOT break existing variants. Each test freezes the
// exact makeSpec() call from a .hip variant file and asserts on the
// full GemmSpec output.
//
// If a test fails after a schema change, the change is NOT backwards-
// compatible. Fix the schema or update the variant (and document why).

#include <rocm_ck/gemm_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::AddOp;
using ::rocm_ck::DataType;
using ::rocm_ck::EpilogueOp;
using ::rocm_ck::GemmAlgorithm;
using ::rocm_ck::GemmOp;
using ::rocm_ck::GemmSpec;
using ::rocm_ck::Layout;
using ::rocm_ck::makeSpec;
using ::rocm_ck::ReluOp;
using ::rocm_ck::Signature;
using ::rocm_ck::TargetSet;

// Frozen baseline tests: these assert ALL fields of each spec variant.
// This is intentionally brittle -- adding a new field to GemmSpec will
// break these tests, forcing explicit review of the change's impact on
// existing variants. Update the expected values when making intentional
// schema changes.

// ============================================================================
// gemm_fp32: FP32 plain GEMM, 16x16x16 MFMA tile
// ============================================================================

TEST(SchemaCompat, GemmFP32)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{
            .block_tile = {128, 128, 32}, .block_waves = {2, 2, 1}, .wave_tile = {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 3);
    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "C"), 2);
    EXPECT_EQ(dtype(k, "A"), DataType::FP32);
    EXPECT_EQ(dtype(k, "B"), DataType::FP32);
    EXPECT_EQ(dtype(k, "C"), DataType::FP32);
    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
    EXPECT_EQ(k.num_epilogue_ops, 0);
    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.wave_tile.m, 16);
    EXPECT_EQ(k.wave_tile.n, 16);
    EXPECT_EQ(k.wave_tile.k, 16);
}

// ============================================================================
// gemm_fp16: FP16 plain GEMM, 16x16x16 MFMA tile
// ============================================================================

TEST(SchemaCompat, GemmFP16)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{
            .block_tile = {128, 128, 32}, .block_waves = {2, 2, 1}, .wave_tile = {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 3);
    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "C"), 2);
    EXPECT_EQ(dtype(k, "A"), DataType::FP16);
    EXPECT_EQ(dtype(k, "B"), DataType::FP16);
    EXPECT_EQ(dtype(k, "C"), DataType::FP16);
    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
    EXPECT_EQ(k.num_epilogue_ops, 0);
    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.wave_tile.m, 16);
    EXPECT_EQ(k.wave_tile.n, 16);
    EXPECT_EQ(k.wave_tile.k, 16);
}

// ============================================================================
// gemm_bf16: BF16 plain GEMM, 16x16x16 MFMA tile
// ============================================================================

TEST(SchemaCompat, GemmBF16)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::BF16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{
            .block_tile = {128, 128, 32}, .block_waves = {2, 2, 1}, .wave_tile = {16, 16, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 3);
    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "C"), 2);
    EXPECT_EQ(dtype(k, "A"), DataType::BF16);
    EXPECT_EQ(dtype(k, "B"), DataType::BF16);
    EXPECT_EQ(dtype(k, "C"), DataType::BF16);
    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
    EXPECT_EQ(k.num_epilogue_ops, 0);
    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.wave_tile.m, 16);
    EXPECT_EQ(k.wave_tile.n, 16);
    EXPECT_EQ(k.wave_tile.k, 16);
}

// ============================================================================
// gemm_fp16_w32: FP16 plain GEMM, 32x32x16 MFMA tile
// ============================================================================

TEST(SchemaCompat, GemmFP16W32)
{
    constexpr auto k = makeSpec(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
        GemmAlgorithm{
            .block_tile = {128, 128, 32}, .block_waves = {2, 2, 1}, .wave_tile = {32, 32, 16}},
        TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 3);
    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "C"), 2);
    EXPECT_EQ(dtype(k, "A"), DataType::FP16);
    EXPECT_EQ(dtype(k, "B"), DataType::FP16);
    EXPECT_EQ(dtype(k, "C"), DataType::FP16);
    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(layout(k, "C"), Layout::Row);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
    EXPECT_EQ(k.num_epilogue_ops, 0);
    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.wave_tile.m, 32);
    EXPECT_EQ(k.wave_tile.n, 32);
    EXPECT_EQ(k.wave_tile.k, 16);
}

// ============================================================================
// gemm_fp16_add: FP16 GEMM + Add (1 D tensor)
// ============================================================================

TEST(SchemaCompat, GemmFP16Add)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}},
                                GemmAlgorithm{.block_tile  = {128, 128, 32},
                                              .block_waves = {2, 2, 1},
                                              .wave_tile   = {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 4);
    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "D"), 2);    // final output
    EXPECT_EQ(slot(k, "bias"), 3); // D0 slot
    EXPECT_EQ(dtype(k, "A"), DataType::FP16);
    EXPECT_EQ(dtype(k, "B"), DataType::FP16);
    EXPECT_EQ(dtype(k, "D"), DataType::FP16);
    EXPECT_EQ(dtype(k, "bias"), DataType::FP16);
    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
    EXPECT_EQ(k.num_epilogue_ops, 1);
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Add));
    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.wave_tile.m, 16);
    EXPECT_EQ(k.wave_tile.n, 16);
    EXPECT_EQ(k.wave_tile.k, 16);
}

// ============================================================================
// gemm_fp16_add_relu: FP16 GEMM + Add + Relu (1 D tensor)
// ============================================================================

TEST(SchemaCompat, GemmFP16AddRelu)
{
    constexpr auto k = makeSpec(Signature{.dtype = DataType::FP16,
                                          .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                    AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                    ReluOp{.in = "D", .out = "E"}}},
                                GemmAlgorithm{.block_tile  = {128, 128, 32},
                                              .block_waves = {2, 2, 1},
                                              .wave_tile   = {16, 16, 16}},
                                TargetSet::cdna());

    EXPECT_EQ(k.num_physical_tensors, 4);
    EXPECT_EQ(slot(k, "A"), 0);
    EXPECT_EQ(slot(k, "B"), 1);
    EXPECT_EQ(slot(k, "E"), 2);    // final output
    EXPECT_EQ(slot(k, "bias"), 3); // D0 slot
    EXPECT_EQ(dtype(k, "A"), DataType::FP16);
    EXPECT_EQ(dtype(k, "B"), DataType::FP16);
    EXPECT_EQ(dtype(k, "E"), DataType::FP16);
    EXPECT_EQ(dtype(k, "bias"), DataType::FP16);
    EXPECT_EQ(layout(k, "A"), Layout::Row);
    EXPECT_EQ(layout(k, "B"), Layout::Col);
    EXPECT_EQ(k.acc_dtype, DataType::FP32);
    EXPECT_EQ(k.num_epilogue_ops, 2);
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Add));
    EXPECT_TRUE(k.hasEpilogueOp(EpilogueOp::Relu));
    EXPECT_EQ(k.workgroup_size, 256);
    EXPECT_EQ(k.wave_tile.m, 16);
    EXPECT_EQ(k.wave_tile.n, 16);
    EXPECT_EQ(k.wave_tile.k, 16);
}
