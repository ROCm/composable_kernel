// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: GROUP mode with pad_hdim_q=0 AND pad_hdim_v=0.
// Expected error: "group mode requires padding"

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::GROUP},
    .algorithm = {.pad_hdim_q = 0, .pad_hdim_v = 0}});
