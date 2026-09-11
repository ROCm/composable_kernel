// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_per_cu=-1 is not positive (no auto-resolution for OGradDotO).
// Expected error: "block_per_cu must be positive"

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true, .block_per_cu = -1}});
