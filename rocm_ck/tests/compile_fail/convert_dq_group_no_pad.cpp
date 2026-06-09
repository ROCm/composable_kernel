// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: GROUP mode requires pad_seqlen_q=true.
// Expected error: "group mode requires pad_seqlen_q=true"

#include <rocm_ck/ops/fmha_bwd/convert_dq_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdConvertDQConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128, .mode = FmhaMode::GROUP},
    .algorithm = {.pad_seqlen_q = false}});
