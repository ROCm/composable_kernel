// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: asymmetric head dimensions (hdim_q != hdim_v) are unsupported.
// CK Tile's fmha_bwd.py defines tuned dQ/dK/dV tile configs only for symmetric
// head dims, so getTileConfig() rejects this combination at compile time.
// Expected error: "requires hdim_q == hdim_v"
//
// NOTE: the compile_fail harness only asserts the TU fails to build (WILL_FAIL);
// it does not match this message. The (hdim_q, hdim_v) pair below is chosen so
// both values are individually valid (each in {32,64,96,128,256}), making the
// asymmetric throw the only reachable failure rather than an earlier range guard.

#include <rocm_ck/ops/fmha_bwd/dqdkdv_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = makeSpec(FmhaBwdDQDKDVConfig{
    .signature = {.dtype = DataType::FP16, .hdim_q = 64, .hdim_v = 128, .mode = FmhaMode::BATCH},
    .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}});
