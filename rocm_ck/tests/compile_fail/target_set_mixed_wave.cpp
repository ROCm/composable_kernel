// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Expected failure: wavefront_size() on a mixed CDNA+RDNA TargetSet.
// TargetSet::all() contains both wave64 (CDNA) and wave32 (RDNA) targets.
// Calling wavefront_size() on a mixed set must fail at compile time.

#include <rocm_ck/arch_properties.hpp>

using namespace rocm_ck;

constexpr auto mixed   = TargetSet::all();
constexpr int bad_wave = mixed.wavefront_size(); // Must fail: mixed wave64/wave32
