// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: hdim_q=100 is not in {32, 64, 96, 128, 256}.
// Expected error: "hdim_q must be one of {32, 64, 96, 128, 256}"

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 100, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
