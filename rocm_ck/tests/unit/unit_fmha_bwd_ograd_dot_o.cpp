// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_api.hpp>
#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::DataType;
using ::rocm_ck::FmhaBwdOGradDotOAlgorithm;
using ::rocm_ck::FmhaBwdOGradDotOConfig;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::makeSpec;
using ::rocm_ck::ograd_dot_o_grid_size;
namespace S = ::rocm_ck::fmha_bwd_ograd_dot_o_slots;

// ============================================================================
// Algorithm defaults
// ============================================================================

TEST(FmhaBwdOGradDotO, AlgorithmDefaults)
{
    constexpr FmhaBwdOGradDotOAlgorithm algo{.pad_seqlen_q = true, .pad_hdim_v = true};
    EXPECT_EQ(algo.block_per_cu, 2);
    EXPECT_EQ(algo.block_size, 64);
}

// ============================================================================
// makeSpec happy path
// ============================================================================

TEST(FmhaBwdOGradDotO, MakeSpecFP16Batch)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

TEST(FmhaBwdOGradDotO, MakeSpecBF16)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::BF16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(k.dtype, DataType::BF16);
}

TEST(FmhaBwdOGradDotO, MakeSpecGroupMode)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.pad_seqlen_q);
}

TEST(FmhaBwdOGradDotO, MakeSpecAllHdims)
{
    // Test all supported hdim_v values. Each must be a separate constexpr
    // variable because makeSpec is consteval -- cannot use a runtime loop.
    constexpr auto k32  = makeSpec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 32, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k64  = makeSpec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 64, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k96  = makeSpec(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 96, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k128 = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k256 = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 256, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k32.hdim_v, 32);
    EXPECT_EQ(k64.hdim_v, 64);
    EXPECT_EQ(k96.hdim_v, 96);
    EXPECT_EQ(k128.hdim_v, 128);
    EXPECT_EQ(k256.hdim_v, 256);
}

TEST(FmhaBwdOGradDotO, MakeSpecNoPadBatch)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}});
    EXPECT_FALSE(k.pad_seqlen_q);
    EXPECT_FALSE(k.pad_hdim_v);
}

TEST(FmhaBwdOGradDotO, MakeSpecCustomBlockPerCu)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true, .block_per_cu = 4}});
    EXPECT_EQ(k.block_per_cu, 4);
}

TEST(FmhaBwdOGradDotO, MakeSpecCustomBlockSize)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true, .block_size = 128}});
    EXPECT_EQ(k.block_size, 128);
}

// ============================================================================
// Slot constants
// ============================================================================

TEST(FmhaBwdOGradDotO, TensorSlotIndicesAreSequential)
{
    EXPECT_EQ(S::O, 0);
    EXPECT_EQ(S::DO, 1);
    EXPECT_EQ(S::D, 2);
    EXPECT_EQ(S::SEQSTART_Q, 3);
    EXPECT_EQ(S::SEQLEN_Q, 4);
}

TEST(FmhaBwdOGradDotO, ScalarSlotIndices) { EXPECT_EQ(S::P_UNDROP, 0); }

TEST(FmhaBwdOGradDotO, RequiredTensorsBatch)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(S::requiredTensors(k), 3);
}

TEST(FmhaBwdOGradDotO, RequiredTensorsGroup)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(S::requiredTensors(k), 5);
}

TEST(FmhaBwdOGradDotO, RequiredScalarsAlways1)
{
    constexpr auto k_batch = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    constexpr auto k_group = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
    EXPECT_EQ(S::requiredScalars(k_batch), 1);
    EXPECT_EQ(S::requiredScalars(k_group), 1);
}

// ============================================================================
// Grid size
// ============================================================================

TEST(FmhaBwdOGradDotO, GridSizeBasic)
{
    const auto g =
        ograd_dot_o_grid_size(/*batch=*/2, /*nhead=*/8, /*seqlen_q=*/256, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 4u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdOGradDotO, GridSizeCeil)
{
    const auto g = ograd_dot_o_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_q=*/65, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 2u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}

TEST(FmhaBwdOGradDotO, GridSizeExact)
{
    const auto g = ograd_dot_o_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_q=*/64, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 1u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}

// ============================================================================
// Grid size: multi-tile coverage
// ============================================================================

TEST(FmhaBwdOGradDotO, GridSizeMultiTileExact)
{
    // Exact divide: no partial-last-tile path.
    const auto g =
        ograd_dot_o_grid_size(/*batch=*/2, /*nhead=*/8, /*seqlen_q=*/512, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 8u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdOGradDotO, GridSizeMultiTilePartialLastMin)
{
    // Ceil at N*tile + 1 edge: partial last tile covers 1 element.
    const auto g =
        ograd_dot_o_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_q=*/257, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 5u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}

TEST(FmhaBwdOGradDotO, GridSizeMultiTilePartialLastMax)
{
    // Ceil at (N+1)*tile - 1 edge: partial last tile covers tile-1 elements.
    const auto g =
        ograd_dot_o_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_q=*/319, /*tile_m0=*/64);
    EXPECT_EQ(g.x, 5u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}
