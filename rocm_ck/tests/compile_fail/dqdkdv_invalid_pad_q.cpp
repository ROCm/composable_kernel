// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: pad_hdim_q=4 is not in {0, 1, 8}.
// Expected error: "pad_hdim_q must be 0, 1, or 8"

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_hdim_q = 4, .pad_hdim_v = 8}});
