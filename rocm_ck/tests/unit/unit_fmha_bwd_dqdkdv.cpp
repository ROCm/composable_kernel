// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/ops/fmha_bwd/dqdkdv_api.hpp>
#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::DataType;
using ::rocm_ck::dqdkdv_grid_size;
using ::rocm_ck::FmhaBiasType;
using ::rocm_ck::FmhaBwdDQDKDVAlgorithm;
using ::rocm_ck::FmhaBwdDQDKDVConfig;
using ::rocm_ck::FmhaMaskType;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::makeSpec;
namespace S = ::rocm_ck::fmha_bwd_dqdkdv_slots;

// ============================================================================
// Algorithm defaults
// ============================================================================

TEST(FmhaBwdDqDkDv, AlgorithmDefaults)
{
    constexpr FmhaBwdDQDKDVAlgorithm algo{};
    EXPECT_EQ(algo.bias_type, FmhaBiasType::NONE);
    EXPECT_FALSE(algo.has_bias_grad);
    EXPECT_EQ(algo.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(algo.has_dropout);
    EXPECT_FALSE(algo.is_deterministic);
    EXPECT_EQ(algo.pad_hdim_q, 0);
    EXPECT_EQ(algo.pad_hdim_v, 0);
    EXPECT_EQ(algo.block_per_cu, -1);
}

// ============================================================================
// makeSpec happy path
// ============================================================================

TEST(FmhaBwdDqDkDv, MakeSpecBaseline)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::NONE);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1); // auto-resolved from -1
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdDqDkDv, MakeSpecBF16)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::BF16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(k.dtype, DataType::BF16);
}

TEST(FmhaBwdDqDkDv, MakeSpecAllHdimsQ)
{
    // Test all supported hdim_q values (symmetric: hdim_q == hdim_v required).
    // Each must be a separate constexpr variable because makeSpec is consteval.
    constexpr auto k32 = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 32, .hdim_v = 32, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k64 = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64, .hdim_v = 64, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k96 = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 96, .hdim_v = 96, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k128 =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k256 =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 256,
                                                   .hdim_v = 256,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k32.hdim_q, 32);
    EXPECT_EQ(k64.hdim_q, 64);
    EXPECT_EQ(k96.hdim_q, 96);
    EXPECT_EQ(k128.hdim_q, 128);
    EXPECT_EQ(k256.hdim_q, 256);
}

TEST(FmhaBwdDqDkDv, MakeSpecAllHdimsV)
{
    // Test all supported hdim_v values (symmetric: hdim_q == hdim_v required).
    // Each must be a separate constexpr variable because makeSpec is consteval.
    constexpr auto k32 = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 32, .hdim_v = 32, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k64 = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64, .hdim_v = 64, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k96 = makeSpec(FmhaBwdDQDKDVConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 96, .hdim_v = 96, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k128 =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k256 =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 256,
                                                   .hdim_v = 256,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k32.hdim_v, 32);
    EXPECT_EQ(k64.hdim_v, 64);
    EXPECT_EQ(k96.hdim_v, 96);
    EXPECT_EQ(k128.hdim_v, 128);
    EXPECT_EQ(k256.hdim_v, 256);
}

TEST(FmhaBwdDqDkDv, MakeSpecWithMask)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
}

TEST(FmhaBwdDqDkDv, MakeSpecWithDropout)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_TRUE(k.has_dropout);
}

TEST(FmhaBwdDqDkDv, MakeSpecDeterministic)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_TRUE(k.is_deterministic);
}

TEST(FmhaBwdDqDkDv, MakeSpecBiasElementwise)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
}

TEST(FmhaBwdDqDkDv, MakeSpecBiasAlibi)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ALIBI, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(k.bias_type, FmhaBiasType::ALIBI);
}

TEST(FmhaBwdDqDkDv, MakeSpecBiasGrad)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                   .has_bias_grad = true,
                                                   .pad_hdim_q    = 8,
                                                   .pad_hdim_v    = 8}});
    EXPECT_TRUE(k.has_bias_grad);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
}

TEST(FmhaBwdDqDkDv, MakeSpecGroupMode)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
}

TEST(FmhaBwdDqDkDv, MakeSpecGroupModePartialPadQ)
{
    // GROUP with pad_hdim_q=8, pad_hdim_v=0 is valid (AND condition)
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 0}});
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 0);
}

TEST(FmhaBwdDqDkDv, MakeSpecGroupModePartialPadV)
{
    // GROUP with pad_hdim_q=0, pad_hdim_v=8 is valid (AND condition)
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.pad_hdim_q = 0, .pad_hdim_v = 8}});
    EXPECT_EQ(k.pad_hdim_q, 0);
    EXPECT_EQ(k.pad_hdim_v, 8);
}

TEST(FmhaBwdDqDkDv, MakeSpecAllPadValues)
{
    // All valid pad values: 0, 1, 8
    constexpr auto k0 =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 0, .pad_hdim_v = 0}});
    constexpr auto k1 =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 1, .pad_hdim_v = 1}});
    constexpr auto k8 =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k0.pad_hdim_q, 0);
    EXPECT_EQ(k1.pad_hdim_q, 1);
    EXPECT_EQ(k8.pad_hdim_q, 8);
}

TEST(FmhaBwdDqDkDv, MakeSpecExplicitBlockPerCu)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8, .block_per_cu = 2}});
    EXPECT_EQ(k.block_per_cu, 2);
}

TEST(FmhaBwdDqDkDv, BlockPerCuAutoResolvesToOne)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}}); // block_per_cu defaults to -1
    EXPECT_EQ(k.block_per_cu, 1);
}

// ============================================================================
// Slot constants
// ============================================================================

TEST(FmhaBwdDqDkDv, TensorSlotIndicesFixed)
{
    EXPECT_EQ(S::Q, 0);
    EXPECT_EQ(S::K, 1);
    EXPECT_EQ(S::V, 2);
    EXPECT_EQ(S::LSE, 3);
    EXPECT_EQ(S::DO, 4);
    EXPECT_EQ(S::D, 5);
    EXPECT_EQ(S::DQ_ACC, 6);
    EXPECT_EQ(S::DK, 7);
    EXPECT_EQ(S::DV, 8);
    EXPECT_EQ(S::BIAS, 9);
    EXPECT_EQ(S::DBIAS, 10);
    EXPECT_EQ(S::RANDVAL, 11);
}

TEST(FmhaBwdDqDkDv, ScalarSlotIndicesFixed)
{
    EXPECT_EQ(S::RAW_SCALE, 0);
    EXPECT_EQ(S::SCALE, 1);
    EXPECT_EQ(S::NUM_HEAD_Q, 2);
    EXPECT_EQ(S::NHEAD_RATIO_QK, 3);
    EXPECT_EQ(S::P_UNDROP, 4);
    EXPECT_EQ(S::RP_UNDROP, 5);
    EXPECT_EQ(S::DROP_SEED, 6);
    EXPECT_EQ(S::DROP_OFFSET, 7);
    EXPECT_EQ(S::WINDOW_SIZE_LEFT, 8);
    EXPECT_EQ(S::WINDOW_SIZE_RIGHT, 9);
    EXPECT_EQ(S::MASK_TYPE, 10);
}

TEST(FmhaBwdDqDkDv, RequiredScalarsWithMask)
{
    // mask_type alone: needs WINDOW_SIZE_LEFT/RIGHT + MASK_TYPE (slots 8..10) -> 11
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredScalars(k), 11);
}

TEST(FmhaBwdDqDkDv, RequiredScalarsWithMaskAndDropout)
{
    // mask_type + has_dropout: mask slots (8..10) are highest -> 11
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.mask_type   = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .has_dropout = true,
                                                   .pad_hdim_q  = 8,
                                                   .pad_hdim_v  = 8}});
    EXPECT_EQ(S::requiredScalars(k), 11);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsPlain)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredTensors(k), 9);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsWithBias)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                   .has_bias_grad = false,
                                                   .pad_hdim_q    = 8,
                                                   .pad_hdim_v    = 8}});
    EXPECT_EQ(S::requiredTensors(k), 10);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsWithBiasGrad)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                   .has_bias_grad = true,
                                                   .pad_hdim_q    = 8,
                                                   .pad_hdim_v    = 8}});
    EXPECT_EQ(S::requiredTensors(k), 11);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsWithDropout)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredTensors(k), 12);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsWithMask)
{
    // Mask only adds scalar slots (window sizes + mask_type), not tensor slots.
    // Guardrail: a regression that accidentally bumped requiredTensors when
    // mask is enabled would over-allocate tensor entries.
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredTensors(k), 9);
}

TEST(FmhaBwdDqDkDv, RequiredScalarsPlain)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredScalars(k), 4);
}

TEST(FmhaBwdDqDkDv, RequiredScalarsWithDropout)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredScalars(k), 8);
}

TEST(FmhaBwdDqDkDv, RequiredTensorsGroupIncludesSeqSlots)
{
    constexpr auto k_batch =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    constexpr auto k_group =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredTensors(k_batch), 9);
    EXPECT_EQ(S::requiredTensors(k_group), 16);

    // Group mode slot constants
    EXPECT_EQ(S::SEQSTART_Q, 12);
    EXPECT_EQ(S::SEQSTART_K, 13);
    EXPECT_EQ(S::SEQLEN_Q, 14);
    EXPECT_EQ(S::SEQLEN_K, 15);

    // Group mode always returns 16 regardless of features
    constexpr auto k_group_dropout = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
    EXPECT_EQ(S::requiredTensors(k_group_dropout), 16);
}

// ============================================================================
// Grid size
// ============================================================================

TEST(FmhaBwdDqDkDv, GridSizeBasic)
{
    const auto g = dqdkdv_grid_size(/*batch=*/2, /*nhead=*/8, /*seqlen_k=*/256, /*block_n0=*/128);
    EXPECT_EQ(g.x, 2u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdDqDkDv, GridSizeCeil)
{
    const auto g = dqdkdv_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_k=*/129, /*block_n0=*/128);
    EXPECT_EQ(g.x, 2u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}

// ============================================================================
// Grid size: multi-tile coverage
// ============================================================================

TEST(FmhaBwdDqDkDv, GridSizeMultiTileExact)
{
    // Exact divide: no partial-last-tile path.
    const auto g = dqdkdv_grid_size(/*batch=*/2, /*nhead=*/8, /*seqlen_k=*/512, /*block_n0=*/128);
    EXPECT_EQ(g.x, 4u);
    EXPECT_EQ(g.y, 8u);
    EXPECT_EQ(g.z, 2u);
}

TEST(FmhaBwdDqDkDv, GridSizeMultiTilePartialLastMin)
{
    // Ceil at N*tile + 1 edge: partial last tile covers 1 element.
    const auto g = dqdkdv_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_k=*/257, /*block_n0=*/128);
    EXPECT_EQ(g.x, 3u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}

TEST(FmhaBwdDqDkDv, GridSizeMultiTilePartialLastMax)
{
    // Ceil at (N+1)*tile - 1 edge: partial last tile covers tile-1 elements.
    const auto g = dqdkdv_grid_size(/*batch=*/1, /*nhead=*/1, /*seqlen_k=*/383, /*block_n0=*/128);
    EXPECT_EQ(g.x, 3u);
    EXPECT_EQ(g.y, 1u);
    EXPECT_EQ(g.z, 1u);
}
