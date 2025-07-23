// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef __HIPCC_RTC__
#include <string>
#include <string_view>
#include <hip/hip_runtime.h>

namespace ck_tile {

constexpr unsigned int fnv1a_hash(std::string_view str, unsigned int h = 2166136261u)
{
    return str.empty() ? h
                       : fnv1a_hash(str.substr(1),
                                    (h ^ static_cast<unsigned char>(str.front())) * 16777619u);
}
inline std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    const std::string raw_name(props.gcnArchName);
    const auto name = raw_name.substr(0, raw_name.find(':')); // str.substr(0, npos) returns str.
    switch(fnv1a_hash(name))
    {
    // https://github.com/ROCm/MIOpen/blob/8498875aef84878e04c1eabefdf6571514891086/src/target_properties.cpp#L40
    case fnv1a_hash("Ellesmere"):
    case fnv1a_hash("Baffin"):
    case fnv1a_hash("RacerX"):
    case fnv1a_hash("Polaris10"):
    case fnv1a_hash("Polaris11"):
    case fnv1a_hash("Tonga"):
    case fnv1a_hash("Fiji"):
    case fnv1a_hash("gfx800"):
    case fnv1a_hash("gfx802"):
    case fnv1a_hash("gfx804"): return "gfx803";
    case fnv1a_hash("Vega10"):
    case fnv1a_hash("gfx901"): return "gfx900";
    case fnv1a_hash("10.3.0 Sienna_Cichlid 18"): return "gfx1030";
    default: return name;
    }
}
} // namespace ck_tile

#endif
