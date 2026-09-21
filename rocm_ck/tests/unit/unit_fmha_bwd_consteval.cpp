// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Consteval tests for sliding-window attention (SWA) and bottom-right causal
// mask configurations.
//
// These tests verify compile-time guarantees for window_size_left/right and
// mask_type slot layouts, and that the SWA/CMaskBR variant specs are correctly
// derived from the underlying cmask configuration.
//
// NOTE: The SWA/CMaskBR/CMask variants share the same compiled kernel binary.
// The mask_type and window sizes are runtime-parametrized via scalar slots.
// Consteval tests verify the slot infrastructure, not the runtime values.

#include "fmha_bwd_variants.hpp"

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

#include <gtest/gtest.h>

using ::rocm_ck::DataType;
using ::rocm_ck::FmhaBiasType;
using ::rocm_ck::FmhaBwdDQDKDVConfig;
using ::rocm_ck::FmhaBwdDQDKDVSpec;
using ::rocm_ck::FmhaMaskType;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::makeSpec;
using ::rocm_ck::usesBatchSizeSlot;
namespace S = ::rocm_ck::fmha_bwd_dqdkdv_slots;

// Pin the scalar-slot layout for the SWA / mask infrastructure. The runtime
// argument-packing code in the .hip variants writes window sizes and mask_type
// into these exact indices; if anyone renumbers the slots, the build must fail
// here rather than silently corrupting kernel inputs.
static_assert(S::WINDOW_SIZE_LEFT == 8, "WINDOW_SIZE_LEFT slot index drifted");
static_assert(S::WINDOW_SIZE_RIGHT == 9, "WINDOW_SIZE_RIGHT slot index drifted");
static_assert(S::MASK_TYPE == 10, "MASK_TYPE slot index drifted");

// ============================================================================
// Spec equivalence: SWA / CMaskBR / CMask share compiled spec (all fields)
// ============================================================================

TEST(FmhaBwdConsteval, SWA_AllFieldsMatchCMask)
{
    // SWA, CMaskBR, and CMask produce identical compiled specs.
    // The difference is purely runtime (mask_type + window sizes).
    constexpr auto k_cmask =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask");
    constexpr auto k_swa =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_swa");

    // Every field must match -- these share a compiled kernel binary.
    EXPECT_EQ(k_swa.dtype, k_cmask.dtype);
    EXPECT_EQ(k_swa.hdim_q, k_cmask.hdim_q);
    EXPECT_EQ(k_swa.hdim_v, k_cmask.hdim_v);
    EXPECT_EQ(k_swa.mode, k_cmask.mode);
    EXPECT_EQ(k_swa.bias_type, k_cmask.bias_type);
    EXPECT_EQ(k_swa.has_bias_grad, k_cmask.has_bias_grad);
    EXPECT_EQ(k_swa.mask_type, k_cmask.mask_type);
    EXPECT_EQ(k_swa.has_dropout, k_cmask.has_dropout);
    EXPECT_EQ(k_swa.is_deterministic, k_cmask.is_deterministic);
    EXPECT_EQ(k_swa.pad_hdim_q, k_cmask.pad_hdim_q);
    EXPECT_EQ(k_swa.pad_hdim_v, k_cmask.pad_hdim_v);
    EXPECT_EQ(k_swa.block_per_cu, k_cmask.block_per_cu);
    EXPECT_EQ(k_swa.block_size, k_cmask.block_size);
    EXPECT_EQ(k_swa.block_n0, k_cmask.block_n0);
}

TEST(FmhaBwdConsteval, CMaskBR_AllFieldsMatchCMask)
{
    constexpr auto k_cmask =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask");
    constexpr auto k_br =
        ::rocm_ck::fmha_bwd_dqdkdv_variant_spec("fmha_bwd_dqdkdv_fp16_d128_batch_cmask_br");

    EXPECT_EQ(k_br.dtype, k_cmask.dtype);
    EXPECT_EQ(k_br.hdim_q, k_cmask.hdim_q);
    EXPECT_EQ(k_br.hdim_v, k_cmask.hdim_v);
    EXPECT_EQ(k_br.mode, k_cmask.mode);
    EXPECT_EQ(k_br.bias_type, k_cmask.bias_type);
    EXPECT_EQ(k_br.has_bias_grad, k_cmask.has_bias_grad);
    EXPECT_EQ(k_br.mask_type, k_cmask.mask_type);
    EXPECT_EQ(k_br.has_dropout, k_cmask.has_dropout);
    EXPECT_EQ(k_br.is_deterministic, k_cmask.is_deterministic);
    EXPECT_EQ(k_br.pad_hdim_q, k_cmask.pad_hdim_q);
    EXPECT_EQ(k_br.pad_hdim_v, k_cmask.pad_hdim_v);
    EXPECT_EQ(k_br.block_per_cu, k_cmask.block_per_cu);
    EXPECT_EQ(k_br.block_size, k_cmask.block_size);
    EXPECT_EQ(k_br.block_n0, k_cmask.block_n0);
}

// ============================================================================
// Spec generation with mask: scalar slot requirements
// (These test the underlying makeSpec() system. The SWA/CMaskBR/CMask variants
//  all use this same spec infrastructure.)
// ============================================================================

TEST(FmhaBwdConsteval, MaskedSpec_RequiredScalars_DeterministicBatch)
{
    // Deterministic batch mode adds BATCH_SIZE slot (index 11) -> 12 scalars.
    // This dominates the mask slots (index 10) -> 11 scalars.
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_EQ(S::requiredScalars(k), 12);
}

TEST(FmhaBwdConsteval, MaskedSpec_RequiredScalars_DeterministicGroup)
{
    // Group mode does not use the BATCH_SIZE slot (kernel derives batch count
    // from seqstart pointers), so mask slots dominate -> 11 scalars.
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_EQ(S::requiredScalars(k), 11);
}

// ============================================================================
// usesBatchSizeSlot with mask combinations
// ============================================================================

TEST(FmhaBwdConsteval, MaskedSpec_UsesBatchSizeSlot_WhenDeterministicBatch)
{
    // Deterministic + batch mode -> uses BATCH_SIZE slot.
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_TRUE(usesBatchSizeSlot(k));
}

TEST(FmhaBwdConsteval, MaskedSpec_DoesNotUseBatchSizeSlot_WhenDeterministicGroup)
{
    // Deterministic + group mode -> does NOT use BATCH_SIZE slot.
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::FP16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::GROUP},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_FALSE(usesBatchSizeSlot(k));
}

TEST(FmhaBwdConsteval, MaskedSpec_DoesNotUseBatchSizeSlot_WhenNonDeterministic)
{
    // Mask without deterministic flag: batch slot is NOT required.
    // Only mask+deterministic+batch uses the batch slot.
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_FALSE(usesBatchSizeSlot(k));
}

// ============================================================================
// Tensor slot invariance: mask does not add tensor slots
// ============================================================================

TEST(FmhaBwdConsteval, MaskedSpec_RequiredTensors_UnchangedForGroup)
{
    // Group mode always requires 16 tensor slots, regardless of mask.
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(S::requiredTensors(k), 16);
}

// ============================================================================
// BF16 mask spec: SWA configurations work with both dtypes
// ============================================================================

TEST(FmhaBwdConsteval, MaskedSpec_BF16_WithMask)
{
    constexpr auto k = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::BF16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {
            .mask_type = FmhaMaskType::TOP_LEFT_CAUSAL, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_EQ(S::requiredScalars(k), 11);
}

TEST(FmhaBwdConsteval, MaskedSpec_BF16_WithMaskAndDeterministic)
{
    constexpr auto k =
        makeSpec(FmhaBwdDQDKDVConfig{.signature = {.dtype  = DataType::BF16,
                                                   .hdim_q = 128,
                                                   .hdim_v = 128,
                                                   .mode   = FmhaMode::BATCH},
                                     .algorithm = {.mask_type = FmhaMaskType::TOP_LEFT_CAUSAL,
                                                   .is_deterministic = true,
                                                   .pad_hdim_q       = 8,
                                                   .pad_hdim_v       = 8}});

    EXPECT_EQ(k.dtype, DataType::BF16);
    EXPECT_NE(k.mask_type, FmhaMaskType::NO_MASK);
    EXPECT_TRUE(k.is_deterministic);
    EXPECT_EQ(S::requiredScalars(k), 12);
}
