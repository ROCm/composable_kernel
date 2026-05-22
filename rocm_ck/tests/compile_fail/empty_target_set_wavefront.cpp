// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: wavefront_size() called on empty TargetSet.
// Expected error: "wavefront_size() called on empty TargetSet"

#include <rocm_ck/arch_properties.hpp>

using namespace rocm_ck;

constexpr int bad = TargetSet{}.wavefront_size();
