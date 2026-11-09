// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

// Runtime SiLU override: clamp BEFORE SiLU, with no alpha or +1.
struct MoeSilu
{
    __host__ __device__ static bool ValidLimit(float limit) { return limit >= 0.0f; }

    __host__ __device__ static bool ValidRuntimeLimit(float limit)
    {
        return limit > 0.0f && limit < __builtin_inff();
    }

    __host__ __device__ static float Run(float gate, float up, float limit)
    {
        gate = math::min(gate, limit);
        up   = math::min(math::max(up, -limit), limit);
        tensor_operation::element_wise::Silu{}(gate, gate);
        return gate * up;
    }

    // swiglustep: SiLU first, then clamp at 7. Never a runtime override.
    __host__ __device__ static float Swiglustep(float gate, float up)
    {
        tensor_operation::element_wise::Silu{}(gate, gate);
        gate = gate < 7.0f ? gate : 7.0f;
        up   = up < 7.0f ? (up > -7.0f ? up : -7.0f) : 7.0f;
        return gate * up;
    }
};

} // namespace ck
