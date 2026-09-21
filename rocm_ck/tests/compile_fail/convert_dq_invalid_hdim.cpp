// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: hdim_q=100 is not in {32, 64, 96, 128, 256}.
// Expected error: "hdim_q must be one of {32, 64, 96, 128, 256}"

#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 100, .mode = FmhaMode::BATCH},
    .algorithm = {}});
