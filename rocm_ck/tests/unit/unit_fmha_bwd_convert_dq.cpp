// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/convert_dq_api.hpp>
#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::convert_dq_grid_size;
using ::rocm_ck::DataType;
using ::rocm_ck::FmhaBwdConvertDQAlgorithm;
using ::rocm_ck::FmhaBwdConvertDQConfig;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::makeSpec;
namespace S = ::rocm_ck::fmha_bwd_convert_dq_slots;

// ============================================================================
// Algorithm defaults
// ============================================================================

TEST(FmhaBwdConvertDQ, AlgorithmDefaults)
{
    constexpr FmhaBwdConvertDQAlgorithm algo{};
    EXPECT_TRUE(algo.is_deterministic);
    EXPECT_TRUE(algo.pad_seqlen_q);
    EXPECT_TRUE(algo.pad_hdim_q);
    EXPECT_EQ(algo.block_per_cu, 2);
}

// ============================================================================
// makeSpec happy path
// ============================================================================

TEST(FmhaBwdConvertDQ, MakeSpecFP16Batch)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_q = true}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

TEST(FmhaBwdConvertDQ, MakeSpecBF16)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::BF16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_EQ(k.dtype, DataType::BF16);
}

TEST(FmhaBwdConvertDQ, MakeSpecGroupMode)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_q = true}});
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.pad_seqlen_q);
}

TEST(FmhaBwdConvertDQ, MakeSpecAllHdims)
{
    // Test all supported hdim_q values. Each must be a separate constexpr
    // variable because makeSpec is consteval -- cannot use a runtime loop.
    constexpr auto k32  = makeSpec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 32, .mode = FmhaMode::BATCH},
         .algorithm = {}});
    constexpr auto k64  = makeSpec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 64, .mode = FmhaMode::BATCH},
         .algorithm = {}});
    constexpr auto k96  = makeSpec(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 96, .mode = FmhaMode::BATCH},
         .algorithm = {}});
    constexpr auto k128 = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    constexpr auto k256 = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 256, .mode = FmhaMode::BATCH},
        .algorithm = {}});

    EXPECT_EQ(k32.hdim_q, 32);
    EXPECT_EQ(k64.hdim_q, 64);
    EXPECT_EQ(k96.hdim_q, 96);
    EXPECT_EQ(k128.hdim_q, 128);
    EXPECT_EQ(k256.hdim_q, 256);
}

TEST(FmhaBwdConvertDQ, MakeSpecNoPadBatch)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_q = false}});
    EXPECT_FALSE(k.pad_seqlen_q);
    EXPECT_FALSE(k.pad_hdim_q);
}

TEST(FmhaBwdConvertDQ, MakeSpecCustomBlockPerCu)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.block_per_cu = 4}});
    EXPECT_EQ(k.block_per_cu, 4);
}

TEST(FmhaBwdConvertDQ, IsDeterministicDefault)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_TRUE(k.is_deterministic);
}

TEST(FmhaBwdConvertDQ, MakeSpecNonDeterministic)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = false}});
    EXPECT_FALSE(k.is_deterministic);
}

// ============================================================================
// Slot constants
// ============================================================================

TEST(FmhaBwdConvertDQ, TensorSlotIndices)
{
    EXPECT_EQ(S::DQ_ACC, 0);
    EXPECT_EQ(S::DQ, 1);
    EXPECT_EQ(S::SEQSTART_Q, 2);
    EXPECT_EQ(S::SEQLEN_Q, 3);
    EXPECT_EQ(S::SEQSTART_K, 4);
    EXPECT_EQ(S::SEQLEN_K, 5);
}

TEST(FmhaBwdConvertDQ, RequiredTensorsBatch)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_EQ(S::requiredTensors(k), 3);
}

TEST(FmhaBwdConvertDQ, RequiredTensorsGroup)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::GROUP},
        .algorithm = {}});
    EXPECT_EQ(S::requiredTensors(k), 8);
}

TEST(FmhaBwdConvertDQ, RequiredScalarsAlways0)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});
    EXPECT_EQ(S::requiredScalars(k), 0);
}

// ============================================================================
// Grid size
// ============================================================================

TEST(FmhaBwdConvertDQ, GridSizeBasic)
{
    const auto g = convert_dq_grid_size(/*batch=*/2, /*nhead=*/8, /*seqlen_q=*/256, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 4u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdConvertDQ, GridSizeDefaultTileM0)
{
    // Default tile_m0 = 64
    const auto g = convert_dq_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_q=*/128);
    EXPECT_EQ(g.x, 2u);
}

// ============================================================================
// Grid size: multi-tile coverage
// ============================================================================

TEST(FmhaBwdConvertDQ, GridSizeMultiTileExact)
{
    // Exact divide: no partial-last-tile path.
    const auto g = convert_dq_grid_size(/*batch=*/2, /*nhead=*/8, /*seqlen_q=*/512, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 8u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdConvertDQ, GridSizeMultiTilePartialLastMin)
{
    // Ceil at N*tile + 1 edge: partial last tile covers 1 element.
    const auto g = convert_dq_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_q=*/257, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 5u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}

TEST(FmhaBwdConvertDQ, GridSizeMultiTilePartialLastMax)
{
    // Ceil at (N+1)*tile - 1 edge: partial last tile covers tile-1 elements.
    const auto g = convert_dq_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_q=*/319, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 5u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}
