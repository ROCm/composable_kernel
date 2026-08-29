// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_size=65 is not divisible by warp size (64).
// Expected error: "block_size must be divisible by warp size (64)"

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true, .block_size = 65}});
