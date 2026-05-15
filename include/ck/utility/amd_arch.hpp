// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include "ck/ck.hpp"

namespace ck {

// Architecture tags
struct gfx9_t
{
};
struct gfx950_t
{
};
struct gfx103_t
{
};
struct gfx11_t
{
};
struct gfx12_t
{
};
struct gfx120_t
{
};
struct gfx125_t
{
};
struct gfx_invalid_t
{
};

static constexpr auto get_device_arch()
{
#if defined(__gfx950__)
    return gfx950_t{};
#elif defined(__gfx9__)
    return gfx9_t{};
#elif defined(__gfx10__)
    return gfx103_t{};
#elif defined(__gfx11__)
    return gfx11_t{};
#elif defined(__gfx125__)
    return gfx125_t{};
#elif defined(__gfx12__)
    return gfx120_t{};
#else
    return gfx_invalid_t{};
#endif
}

template <typename DeviceArch>
static constexpr index_t get_lds_size(DeviceArch)
{
    return 64 * 1024;
}
template <>
constexpr index_t get_lds_size<gfx950_t>(gfx950_t)
{
    return 160 * 1024;
}
template <>
constexpr index_t get_lds_size<gfx125_t>(gfx125_t)
{
    return 320 * 1024;
}

template <typename DeviceArch>
static constexpr index_t get_n_lds_banks(DeviceArch)
{
    return 32;
}
template <>
constexpr index_t get_n_lds_banks<gfx950_t>(gfx950_t)
{
    return 64;
}
template <>
constexpr index_t get_n_lds_banks<gfx125_t>(gfx125_t)
{
    return 64;
}

template <typename DeviceArch>
static constexpr index_t get_max_vgpr_count(DeviceArch)
{
    return 256;
}
template <>
constexpr index_t get_max_vgpr_count<gfx950_t>(gfx950_t)
{
    return 512;
}
template <>
constexpr index_t get_max_vgpr_count<gfx9_t>(gfx9_t)
{
    return 512;
}
template <>
constexpr index_t get_max_vgpr_count<gfx125_t>(gfx125_t)
{
    return 1024;
}

template <typename DeviceArch>
static constexpr index_t get_vgpr_count_per_simd(DeviceArch)
{
    return 1024;
}

template <>
constexpr index_t get_vgpr_count_per_simd<gfx9_t>(gfx9_t)
{
    return 512;
}
template <>
constexpr index_t get_vgpr_count_per_simd<gfx950_t>(gfx950_t)
{
    return 512;
}

} // namespace ck
