// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: FP32 not supported for ConvertDQ.
// Expected error: "only supports FP16 or BF16"

#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP32, .hdim_q = 128, .mode = FmhaMode::BATCH},
    .algorithm = {}});
