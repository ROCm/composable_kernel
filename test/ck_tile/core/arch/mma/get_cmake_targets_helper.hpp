// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include <unordered_set>

namespace ck_tile::core::arch::testing {

static CK_TILE_HOST auto getCMakeGpuTargetIds()
{
    using ck_tile::core::arch::amdgcn_target_id;
#ifdef CK_CMAKE_GPU_TARGET_IDS
    constexpr uint32_t ids[] = {CK_CMAKE_GPU_TARGET_IDS};
    std::unordered_set<amdgcn_target_id> result;
    for(auto id : ids)
        result.insert(static_cast<amdgcn_target_id>(id));
    return result;
#else
    return std::unordered_set<amdgcn_target_id>{};
#endif
}

template <typename Func>
static CK_TILE_HOST bool dispatchCompilerTarget(ck_tile::core::arch::amdgcn_target_id id,
                                                Func&& func)
{
    using namespace ck_tile::core::arch;

    // clang-format off
    switch(id)
    {
    case amdgcn_target_id::GFX908:         func(make_amdgcn_gfx9_target<amdgcn_target_id::GFX908>());            return true;
    case amdgcn_target_id::GFX90A:         func(make_amdgcn_gfx9_target<amdgcn_target_id::GFX90A>());            return true;
    case amdgcn_target_id::GFX942:         func(make_amdgcn_gfx9_target<amdgcn_target_id::GFX942>());            return true;
    case amdgcn_target_id::GFX950:         func(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());            return true;
    case amdgcn_target_id::GFX1030:        func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1030>());        return true;
    case amdgcn_target_id::GFX1031:        func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1031>());        return true;
    case amdgcn_target_id::GFX1032:        func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1032>());        return true;
    case amdgcn_target_id::GFX1033:        func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1033>());        return true;
    case amdgcn_target_id::GFX1034:        func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1034>());        return true;
    case amdgcn_target_id::GFX1035:        func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1035>());        return true;
    case amdgcn_target_id::GFX1036:        func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX1036>());        return true;
    case amdgcn_target_id::GFX103_GENERIC: func(make_amdgcn_gfx10_3_target<amdgcn_target_id::GFX103_GENERIC>()); return true;
    case amdgcn_target_id::GFX1100:        func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>());          return true;
    case amdgcn_target_id::GFX1101:        func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1101>());          return true;
    case amdgcn_target_id::GFX1102:        func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1102>());          return true;
    case amdgcn_target_id::GFX1103:        func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1103>());          return true;
    case amdgcn_target_id::GFX1150:        func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1150>());          return true;
    case amdgcn_target_id::GFX1151:        func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1151>());          return true;
    case amdgcn_target_id::GFX1152:        func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1152>());          return true;
    case amdgcn_target_id::GFX1153:        func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1153>());          return true;
    case amdgcn_target_id::GFX11_GENERIC:  func(make_amdgcn_gfx11_target<amdgcn_target_id::GFX11_GENERIC>());    return true;
    case amdgcn_target_id::GFX1200:        func(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1200>());          return true;
    case amdgcn_target_id::GFX1201:        func(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());          return true;
    case amdgcn_target_id::GFX12_GENERIC:  func(make_amdgcn_gfx12_target<amdgcn_target_id::GFX12_GENERIC>());    return true;
    case amdgcn_target_id::GFX1250:        func(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1250>());          return true;
    case amdgcn_target_id::HOST:           return false;
    }
    // clang-format on
    __builtin_unreachable();
}

static CK_TILE_HOST constexpr int32_t getCMakeWaveSize()
{
    using ck_tile::core::arch::amdgcn_target_id;
#ifdef CK_CMAKE_GPU_TARGET_IDS
    constexpr uint32_t ids[]       = {CK_CMAKE_GPU_TARGET_IDS};
    constexpr index_t targets_size = sizeof(ids) / sizeof(ids[0]);
    static_assert(targets_size > 0);
    constexpr auto first_target_id = static_cast<amdgcn_target_id>(ids[0]);
    if constexpr(first_target_id >= amdgcn_target_id::GFX908 &&
                 first_target_id <= amdgcn_target_id::GFX950)
    {
        return 64;
    }
    else
    {
        return 32;
    }
#else
    static_assert(false, "Configure CK_CMAKE_GPU_TARGET_IDS before calling this function.");
    return 0;
#endif
}
} // namespace ck_tile::core::arch::testing
