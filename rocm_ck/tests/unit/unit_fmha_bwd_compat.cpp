// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Frozen baseline tests for FMHA BWD kernel variants.
//
// Each test reproduces the exact makeSpec() call from the variant table
// and asserts ALL fields. If a schema change breaks a test,
// the change is not backwards-compatible.
//
// NOTE: block_size and block_n0 in DqDkDv are default constants
// (dqdkdv_spec.hpp). Update when architecture-dependent tile
// selection is implemented.

#include "fmha_bwd_variants.hpp"

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>
#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>
#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

#include <gtest/gtest.h>

#include <cstring>

using ::rocm_ck::DataType;
using ::rocm_ck::FmhaBiasType;
using ::rocm_ck::FmhaBwdConvertDQConfig;
using ::rocm_ck::FmhaBwdDQDKDVConfig;
using ::rocm_ck::FmhaBwdOGradDotOConfig;
using ::rocm_ck::FmhaMaskType;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::makeSpec;

// ============================================================================
// OGradDotO frozen baselines
// ============================================================================

TEST(FmhaBwdCompat, OGradDotO_FP16_D128_Batch)
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

TEST(FmhaBwdCompat, OGradDotO_BF16_D128_Batch)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::BF16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

TEST(FmhaBwdCompat, OGradDotO_FP16_D64_Batch)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 64, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_v, 64);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

TEST(FmhaBwdCompat, OGradDotO_FP16_D128_Group)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

TEST(FmhaBwdCompat, OGradDotO_FP16_D128_Batch_NoPad)
{
    constexpr auto k = makeSpec(FmhaBwdOGradDotOConfig{
        .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_FALSE(k.pad_seqlen_q);
    EXPECT_FALSE(k.pad_hdim_v);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 64);
}

// ============================================================================
// DqDkDv frozen baselines
// ============================================================================

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch)
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
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_BF16_D128_Batch)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::BF16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_CMask)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_Det)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_TRUE(k.is_deterministic);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Group)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_EQ(k.block_per_cu, 1);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Group_CMask)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Group_Det)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.is_deterministic = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_TRUE(k.is_deterministic);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Group_Dropout)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_TRUE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Group_EBias)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Group_ALiBi)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {.bias_type = FmhaBiasType::ALIBI, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ALIBI);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Group_EBias_DBias)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                   .has_bias_grad = true,
                                                   .pad_hdim_q    = 8,
                                                   .pad_hdim_v    = 8}});

    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
    EXPECT_TRUE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_EBias)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_ALiBi)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ALIBI, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ALIBI);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_EBias_DBias)
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

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
    EXPECT_TRUE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_BF16_D128_Batch_EBias)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::BF16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ELEMENTWISE, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_BF16_D128_Batch_ALiBi)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::BF16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.bias_type = FmhaBiasType::ALIBI, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ALIBI);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_BF16_D128_Batch_EBias_DBias)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::BF16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.bias_type     = FmhaBiasType::ELEMENTWISE,
                                                   .has_bias_grad = true,
                                                   .pad_hdim_q    = 8,
                                                   .pad_hdim_v    = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::ELEMENTWISE);
    EXPECT_TRUE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_Dropout)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::NONE);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_EQ(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_TRUE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_CMask_Det)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::NONE);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

// Bottom-right causal and sliding-window variants share the compiled spec
// with _cmask. They are separate registry entries, reachable only via
// fmha_bwd_dqdkdv_variant_spec("<exact name>") (consteval name lookup).
// mask_type and window_size_left/right are runtime-parametrized via scalar slots.

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_CMaskBR)
{
    constexpr auto k =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask_br");

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::NONE);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_SWA)
{
    constexpr auto k =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_swa");

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_EQ(k.bias_type, FmhaBiasType::NONE);
    EXPECT_FALSE(k.has_bias_grad);
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_FALSE(k.has_dropout);
    EXPECT_FALSE(k.is_deterministic);
    EXPECT_EQ(k.pad_hdim_q, 8);
    EXPECT_EQ(k.pad_hdim_v, 8);
    EXPECT_EQ(k.block_per_cu, 1);
    EXPECT_EQ(k.block_size, 256);
    EXPECT_EQ(k.block_n0, 128);
}

// ============================================================================
// ConvertDQ frozen baselines
// ============================================================================

TEST(FmhaBwdCompat, ConvertDQ_FP16_D64_Batch)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64, .mode = FmhaMode::BATCH},
        .algorithm = {}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 64);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

TEST(FmhaBwdCompat, ConvertDQ_FP16_D128_Batch)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
        .algorithm = {}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

TEST(FmhaBwdCompat, ConvertDQ_FP16_D128_Group)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::GROUP},
        .algorithm = {}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

TEST(FmhaBwdCompat, ConvertDQ_FP16_D256_Batch)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 256, .mode = FmhaMode::BATCH},
        .algorithm = {}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 256);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

TEST(FmhaBwdCompat, ConvertDQ_FP16_D64_Group)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 64, .mode = FmhaMode::GROUP},
        .algorithm = {}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 64);
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

TEST(FmhaBwdCompat, ConvertDQ_FP16_D256_Group)
{
    constexpr auto k = makeSpec(FmhaBwdConvertDQConfig{
        .signature = {.dtype = DataType::FP16, .hdim_q = 256, .mode = FmhaMode::GROUP},
        .algorithm = {}});

    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 256);
    EXPECT_EQ(k.mode, FmhaMode::GROUP);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_TRUE(k.pad_seqlen_q);
    EXPECT_TRUE(k.pad_hdim_q);
    EXPECT_EQ(k.block_per_cu, 2);
    EXPECT_EQ(k.block_size, 256);
}

// ============================================================================
// Consteval name-based lookup: CMaskBR / SWA share compiled spec with CMask
// ============================================================================

// _cmask_br and _swa share the compiled spec with _cmask. These variants are
// reachable via consteval name lookup (fmha_bwd_dqdkdv_variant_spec) or by
// iterating ALL_DQDKDV_VARIANTS.
TEST(FmhaBwdCompat, Registry_DqDkDv_NameLookup_CMaskBR)
{
    constexpr auto k =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask_br");
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
}

TEST(FmhaBwdCompat, Registry_DqDkDv_NameLookup_SWA)
{
    constexpr auto k =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_swa");
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_EQ(k.dtype, DataType::FP16);
    EXPECT_EQ(k.hdim_q, 128);
    EXPECT_EQ(k.hdim_v, 128);
    EXPECT_EQ(k.mode, FmhaMode::BATCH);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_CMaskBR_SharesSpecWithCMask)
{
    constexpr auto k_cmask =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask");
    constexpr auto k_br =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask_br");

    EXPECT_EQ(k_br.dtype, k_cmask.dtype);
    EXPECT_EQ(k_br.hdim_q, k_cmask.hdim_q);
    EXPECT_EQ(k_br.hdim_v, k_cmask.hdim_v);
    EXPECT_EQ(k_br.mode, k_cmask.mode);
    EXPECT_EQ(k_br.mask_type, k_cmask.mask_type);
    EXPECT_EQ(k_br.pad_hdim_q, k_cmask.pad_hdim_q);
    EXPECT_EQ(k_br.pad_hdim_v, k_cmask.pad_hdim_v);
}

TEST(FmhaBwdCompat, DqDkDv_FP16_D128_Batch_SWA_SharesSpecWithCMask)
{
    constexpr auto k_cmask =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask");
    constexpr auto k_swa =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_swa");

    EXPECT_EQ(k_swa.dtype, k_cmask.dtype);
    EXPECT_EQ(k_swa.hdim_q, k_cmask.hdim_q);
    EXPECT_EQ(k_swa.hdim_v, k_cmask.hdim_v);
    EXPECT_EQ(k_swa.mode, k_cmask.mode);
    EXPECT_EQ(k_swa.mask_type, k_cmask.mask_type);
    EXPECT_EQ(k_swa.pad_hdim_q, k_cmask.pad_hdim_q);
    EXPECT_EQ(k_swa.pad_hdim_v, k_cmask.pad_hdim_v);
}
