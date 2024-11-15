// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/mxfp_utils.hpp"

namespace ck::utils {

__host__ __device__ inline float cast_to_float(e8m0_scale_t const scale)
{
    return std::pow(2, bit_cast<uint8_t>(scale) - NumericUtils<e8m0_scale_t>::bias);
}

__host__ __device__ inline e8m0_scale_t cast_from_float(float const scale)
{
    return static_cast<uint8_t>(std::log2(scale) + NumericUtils<e8m0_scale_t>::bias);
}

template <>
__host__ __device__ inline int get_exponent_value<e8m0_scale_t>(e8m0_scale_t x)
{
    x.data >>= NumericUtils<e8m0_scale_t>::mant;

    x.data &= ((1 << NumericUtils<e8m0_scale_t>::exp) - 1);

    return static_cast<int>(x.data);
}

} // namespace ck::utils
