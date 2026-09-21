// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Death tests for rocm_ck::validateArgs(args, FmhaBwdDQDKDVSpec).
// Verifies that the host-side debug validator aborts on missing tensors and
// on out-of-range scalar values.
//
// validateArgs() body is guarded by #ifndef NDEBUG, so these tests are only
// meaningful in Debug builds. In Release they degenerate to no-ops and the
// EXPECT_DEATH calls would fail; gate them on NDEBUG accordingly.

#include <rocm_ck/ops/fmha_bwd/dqdkdv_api.hpp>

#include <gtest/gtest.h>

#ifndef NDEBUG

using ::rocm_ck::Args;
using ::rocm_ck::DataType;
using ::rocm_ck::FmhaBwdDQDKDVConfig;
using ::rocm_ck::FmhaMode;
using ::rocm_ck::makeShape;
using ::rocm_ck::makeSpec;
using ::rocm_ck::makeStrides;
using ::rocm_ck::validateArgs;
namespace S = ::rocm_ck::fmha_bwd_dqdkdv_slots;

// A minimal valid spec: plain BATCH, no optional features.
static constexpr auto test_spec = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});

// Populate every required tensor slot with a non-null pointer and the two
// sanity-checked scalar slots with valid values. Returns by value so each
// test starts from a clean state.
static Args makeValidArgs()
{
    static int dummy = 1;

    Args args{};
    for(int i : {S::Q, S::K, S::V, S::LSE, S::DO, S::D, S::DQ_ACC, S::DK, S::DV})
    {
        args.tensors[i] = {&dummy, makeShape(1), makeStrides(1)};
    }
    args.scalars[S::SCALE].f32          = 1.0f;
    args.scalars[S::NHEAD_RATIO_QK].i32 = 1;
    // BATCH_SIZE is only required for deterministic+BATCH specs; the default
    // test_spec is non-deterministic so it stays at 0 here. Tests that
    // exercise the deterministic path populate it explicitly.
    return args;
}

// ---------------------------------------------------------------------------
// Pass: a fully populated Args does not abort.
// ---------------------------------------------------------------------------

TEST(FmhaBwdValidateArgs, PassesWhenAllRequiredSlotsFilled)
{
    Args args = makeValidArgs();
    validateArgs(args, test_spec); // must not abort
}

// ---------------------------------------------------------------------------
// Death: each required tensor slot, when null, aborts with the matching name.
// ---------------------------------------------------------------------------

TEST(FmhaBwdValidateArgsDeathTest, AbortsOnNullQ)
{
    Args args              = makeValidArgs();
    args.tensors[S::Q].ptr = nullptr;
    EXPECT_DEATH(validateArgs(args, test_spec), "tensor \"Q\" \\(slot 0\\)");
}

TEST(FmhaBwdValidateArgsDeathTest, AbortsOnNullK)
{
    Args args              = makeValidArgs();
    args.tensors[S::K].ptr = nullptr;
    EXPECT_DEATH(validateArgs(args, test_spec), "tensor \"K\" \\(slot 1\\)");
}

TEST(FmhaBwdValidateArgsDeathTest, AbortsOnNullDV)
{
    Args args               = makeValidArgs();
    args.tensors[S::DV].ptr = nullptr;
    EXPECT_DEATH(validateArgs(args, test_spec), "tensor \"DV\" \\(slot 8\\)");
}

// ---------------------------------------------------------------------------
// Death: invalid scalar values abort.
// ---------------------------------------------------------------------------

TEST(FmhaBwdValidateArgsDeathTest, AbortsOnZeroScale)
{
    Args args                  = makeValidArgs();
    args.scalars[S::SCALE].f32 = 0.0f;
    EXPECT_DEATH(validateArgs(args, test_spec), "SCALE .* is zero");
}

TEST(FmhaBwdValidateArgsDeathTest, AbortsOnNonPositiveNheadRatio)
{
    Args args                           = makeValidArgs();
    args.scalars[S::NHEAD_RATIO_QK].i32 = 0;
    EXPECT_DEATH(validateArgs(args, test_spec), "NHEAD_RATIO_QK .* must be positive");
}

// ---------------------------------------------------------------------------
// Death: deterministic+BATCH spec must populate BATCH_SIZE; non-deterministic
// specs ignore it.
// ---------------------------------------------------------------------------

TEST(FmhaBwdValidateArgsDeathTest, AbortsOnZeroBatchSizeInDeterministicBatch)
{
    constexpr auto det_spec = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.is_deterministic = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    Args args                       = makeValidArgs();
    args.scalars[S::BATCH_SIZE].i32 = 0; // missing/zero -> abort
    EXPECT_DEATH(validateArgs(args, det_spec), "BATCH_SIZE .* must be positive");
}

TEST(FmhaBwdValidateArgs, SkipsBatchSizeCheckForNonDeterministicSpec)
{
    // test_spec is non-deterministic; BATCH_SIZE being zero must NOT abort.
    Args args                       = makeValidArgs();
    args.scalars[S::BATCH_SIZE].i32 = 0;
    validateArgs(args, test_spec);
}

// ---------------------------------------------------------------------------
// Skip rules: optional/feature slots that aren't enabled are not checked.
// ---------------------------------------------------------------------------

TEST(FmhaBwdValidateArgs, SkipsBiasSlotWhenBiasDisabled)
{
    // BIAS slot (9) intentionally null -- spec has bias_type=NONE -> must not abort.
    Args args                 = makeValidArgs();
    args.tensors[S::BIAS].ptr = nullptr;
    validateArgs(args, test_spec);
}

TEST(FmhaBwdValidateArgsDeathTest, AbortsOnNullRandvalWhenDropoutEnabled)
{
    constexpr auto dropout_spec = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    Args args                        = makeValidArgs();
    args.scalars[S::DROP_SEED].u64   = 1;
    args.scalars[S::DROP_OFFSET].u64 = 0;
    args.scalars[S::P_UNDROP].f32    = 1.0f;
    args.scalars[S::RP_UNDROP].f32   = 1.0f;
    args.tensors[S::RANDVAL].ptr     = nullptr;

    EXPECT_DEATH(validateArgs(args, dropout_spec), "tensor \\\"RANDVAL\\\" \\(slot 11\\)");
}

TEST(FmhaBwdValidateArgs, PassesWhenRandvalPresentForDropout)
{
    constexpr auto dropout_spec = makeSpec(FmhaBwdDQDKDVConfig{
        .signature =
            {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
        .algorithm = {.has_dropout = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});

    static int dummy = 1;

    Args args                        = makeValidArgs();
    args.scalars[S::DROP_SEED].u64   = 1;
    args.scalars[S::DROP_OFFSET].u64 = 0;
    args.scalars[S::P_UNDROP].f32    = 1.0f;
    args.scalars[S::RP_UNDROP].f32   = 1.0f;
    args.tensors[S::RANDVAL]         = {&dummy, makeShape(1), makeStrides(1)};

    validateArgs(args, dropout_spec);
}

#endif // NDEBUG
