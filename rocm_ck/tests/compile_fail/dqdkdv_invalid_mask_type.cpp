// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: mask_type must be one of the four declared enum values.
// A value cast in from out-of-range integer space (e.g. 99) is forwarded
// verbatim to the device-side static_cast<ck_tile::GenericAttentionMaskEnum>,
// where it would silently land on undefined kernel behaviour. The consteval
// validator in makeSpec rejects it at compile time instead.
//
// Expected error: "mask_type must be NO_MASK, TOP_LEFT_CAUSAL,
//                  BOTTOM_RIGHT_CAUSAL, or GENERIC"

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 128, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.mask_type = static_cast<FmhaMaskType>(99), .pad_hdim_q = 8, .pad_hdim_v = 8}});
