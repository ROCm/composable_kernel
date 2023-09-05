// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {

template <typename T>
__device__ T warp_shuffle_up(const T& var, uint32_t delta)
{
#if 0
    return  __shfl_up(var, delta);
#elif 1
    const uint32_t wrap_around_delta = warpSize - delta;

    return __builtin_amdgcn_ds_bpermute((__lane_id() << 2) + (wrap_around_delta << 2), var);
#endif
}

template <typename T>
__device__ T warp_shuffle_down(const T& var, uint32_t delta)
{
#if 0
    return  __shfl_down(var, delta);
#elif 1
    return __builtin_amdgcn_ds_bpermute((__lane_id() << 2) + (delta << 2), var);
#endif
}

} // namespace ck
