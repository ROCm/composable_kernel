// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_size=0 is not positive.
// Expected error: "block_size must be positive"

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true, .block_size = 0}});
