// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_per_cu=-2 is not positive (only -1 is auto-resolved).
// Expected error: "block_per_cu must be positive"

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8, .block_per_cu = -2}});
