// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY!
 *
 * Generated from: arch_specs.json
 * Generated at: 2026-06-01T10:50:15.322672
 *
 * To update this file:
 * 1. Edit arch_specs.json
 * 2. Run: python generate_arch_specs.py
 */

#pragma once

#include "ck_tile/dispatcher/kernel_key.hpp"
#include <array>
#include <string>
#include <vector>
#include <cstdint>

namespace ck_tile {
namespace dispatcher {
namespace arch_specs {

// =============================================================================
// GPU Architecture Enum (Generated)
// =============================================================================

enum class GpuArch : std::uint8_t
{
    GFX_908,
    GFX_90A,
    GFX_942,
    GFX_950,
    GFX_1100,
    GFX_1200,
    GFX_1201,
    UNKNOWN
};

// =============================================================================
// String Conversion Functions (Generated)
// =============================================================================

inline std::string arch_to_string(GpuArch arch)
{
    switch(arch)
    {
    case GpuArch::GFX_908: return "gfx908";
    case GpuArch::GFX_90A: return "gfx90a";
    case GpuArch::GFX_942: return "gfx942";
    case GpuArch::GFX_950: return "gfx950";
    case GpuArch::GFX_1100: return "gfx1100";
    case GpuArch::GFX_1200: return "gfx1200";
    case GpuArch::GFX_1201: return "gfx1201";
    default: return "unknown";
    }
}

inline GpuArch string_to_arch(const std::string& arch_str)
{
    if(arch_str == "gfx908")
        return GpuArch::GFX_908;
    if(arch_str == "gfx90a")
        return GpuArch::GFX_90A;
    if(arch_str == "gfx942")
        return GpuArch::GFX_942;
    if(arch_str == "gfx950")
        return GpuArch::GFX_950;
    if(arch_str == "gfx1100")
        return GpuArch::GFX_1100;
    if(arch_str == "gfx1200")
        return GpuArch::GFX_1200;
    if(arch_str == "gfx1201")
        return GpuArch::GFX_1201;
    return GpuArch::UNKNOWN;
}

// =============================================================================
// Element Size (Generated)
// =============================================================================

inline float element_size(DataType dtype)
{
    switch(dtype)
    {
    case DataType::FP16: return 2.0f;
    case DataType::BF16: return 2.0f;
    case DataType::FP32: return 4.0f;
    case DataType::FP64: return 8.0f;
    case DataType::FP8: return 1.0f;
    case DataType::BF8: return 1.0f;
    case DataType::INT8: return 1.0f;
    case DataType::INT4: return 0.5f;
    case DataType::INT32: return 4.0f;
    default: return 2.0f;
    }
}

// =============================================================================
// Warp Configurations (Generated)
// =============================================================================

using WarpConfig = std::array<int, 3>;

inline std::vector<WarpConfig> get_supported_warp_configs(GpuArch arch)
{
    switch(arch)
    {
    case GpuArch::GFX_908: return {{1, 4, 1}, {2, 2, 1}, {4, 1, 1}};
    case GpuArch::GFX_90A: return {{1, 4, 1}, {2, 2, 1}, {4, 1, 1}};
    case GpuArch::GFX_942:
        return {{1, 1, 1}, {1, 2, 1}, {1, 4, 1}, {2, 1, 1}, {2, 1, 2}, {2, 2, 1}, {4, 1, 1}};
    case GpuArch::GFX_950:
        return {{1, 1, 1},
                {1, 2, 1},
                {1, 4, 1},
                {2, 1, 1},
                {2, 1, 2},
                {2, 2, 1},
                {4, 1, 1},
                {8, 2, 1},
                {4, 4, 1}};
    case GpuArch::GFX_1100: return {{2, 4, 1}, {1, 8, 1}, {8, 1, 1}, {4, 2, 1}};
    case GpuArch::GFX_1200: return {{2, 4, 1}, {1, 8, 1}, {8, 1, 1}, {4, 2, 1}};
    case GpuArch::GFX_1201: return {{2, 4, 1}, {1, 8, 1}, {8, 1, 1}, {4, 2, 1}};
    default: return {};
    }
}

// =============================================================================
// LDS Capacity Limits (Generated)
// =============================================================================

inline std::size_t get_lds_capacity(Pipeline pipeline)
{
    if(pipeline == Pipeline::Mem)
        return 65536;
    if(pipeline == Pipeline::CompV1)
        return 65536;
    if(pipeline == Pipeline::CompV2)
        return 65536;
    if(pipeline == Pipeline::CompV3)
        return 65536;
    if(pipeline == Pipeline::CompV4)
        return 32768;
    if(pipeline == Pipeline::CompV5)
        return 65536;
    if(pipeline == Pipeline::CompV6)
        return 32768;
    if(pipeline == Pipeline::PreShuffleV1)
        return 32768;
    if(pipeline == Pipeline::PreShuffleV2)
        return 32768;
    return 65536; // Default
}

// =============================================================================
// Unsupported Trait Combinations (Generated)
// =============================================================================

inline bool
is_trait_unsupported(Pipeline pipeline, [[maybe_unused]] Epilogue epilogue, Scheduler scheduler)
{
    // Generated from unsupported_trait_combos in arch_specs.json
    if(scheduler == Scheduler::Interwave)
    {
        if(pipeline == Pipeline::CompV3 || pipeline == Pipeline::CompV4)
        {
            return true;
        }
    }
    return false;
}

} // namespace arch_specs
} // namespace dispatcher
} // namespace ck_tile
