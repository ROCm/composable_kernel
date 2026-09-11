// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: FP32 not supported for OGradDotO.
// Expected error: "only supports FP16 or BF16"

#include <rocm_ck/ops/fmha_bwd/ograd_dot_o_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdOGradDotOConfig{
    .signature = {.dtype = DataType::FP32, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
