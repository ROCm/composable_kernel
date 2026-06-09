// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: block_per_cu=0 is not positive.
// Expected error: "block_per_cu must be positive"

#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.block_per_cu = 0}});
