// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: has_bias_grad=true requires bias_type != NONE.
// Expected error: "has_bias_grad requires bias_type != NONE"

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.has_bias_grad = true, .pad_hdim_q = 8, .pad_hdim_v = 8}});
