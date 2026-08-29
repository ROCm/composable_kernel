// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: hdim_v=100 is not in {32, 64, 96, 128, 256}.
// Expected error: "hdim_v must be one of {32, 64, 96, 128, 256}"

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP16, .hdim_v = 100, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
