// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile-only: every gfx125 device pass resolves to exactly one target, and gfx1250-strict
// resolves to gfx1250.

#include "ck_tile/core.hpp"

#if defined(__HIP_DEVICE_COMPILE__)
namespace {

using ck_tile::core::arch::amdgcn_target_id;
constexpr auto target_id = ck_tile::core::arch::get_compiler_target().TARGET_ID;

#if defined(__gfx1251__)
static_assert(target_id == amdgcn_target_id::GFX1251, "gfx1251 must resolve to GFX1251");
#elif defined(__gfx1250__) || defined(__gfx1250_strict__)
static_assert(target_id == amdgcn_target_id::GFX1250, "gfx1250(-strict) must resolve to GFX1250");
#else
#error "This file is compiled for the gfx125 targets only"
#endif

} // namespace
#endif
