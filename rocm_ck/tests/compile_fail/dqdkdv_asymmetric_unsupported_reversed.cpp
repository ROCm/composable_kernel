// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: asymmetric head dimensions (hdim_q != hdim_v) are unsupported,
// in either direction. This is the hdim_q > hdim_v companion to
// dqdkdv_asymmetric_unsupported.cpp; getTileConfig() rejects it at compile time.
// Expected error: "requires hdim_q == hdim_v"
//
// NOTE: the compile_fail harness only asserts the TU fails to build (WILL_FAIL);
// it does not match this message. The (hdim_q, hdim_v) pair below is chosen so
// both values are individually valid (each in {32,64,96,128,256}), making the
// asymmetric throw the only reachable failure rather than an earlier range guard.

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 256, .hdim_v = 64, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
