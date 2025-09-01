// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <stdint.h>
#include <tuple>
#include <type_traits>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"

namespace ck_tile {

template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr Y scaled_type_convert(X x, float scale);

#define CK_TILE_SCALED_TYPE_CONVERT(dtype_, dname_, stype_, sname_)                       \
    template <>                                                                           \
    CK_TILE_HOST_DEVICE dtype_ scaled_type_convert<dtype_, stype_>(stype_ x,    \
                                                                             float scale) \
    {                                                                                     \
        return sname_##_to_##dname_(x, scale);                                            \
    }                                                                                     \
    template <>                                                                           \
    CK_TILE_HOST_DEVICE dtype_ type_convert<dtype_, stype_>(stype_ x)           \
    {                                                                                     \
        return sname_##_to_##dname_(x, 1.f);                                              \
    }

CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp32x2_t, fp32x2)
CK_TILE_SCALED_TYPE_CONVERT(fp32x2_t, fp32x2, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp16x2_t, fp16x2)
CK_TILE_SCALED_TYPE_CONVERT(fp16x2_t, fp16x2, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, bf16x2_t, bf16x2)
CK_TILE_SCALED_TYPE_CONVERT(bf16x2_t, bf16x2, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, float, float)
CK_TILE_SCALED_TYPE_CONVERT(float, float, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, bf16_t, bf16)
CK_TILE_SCALED_TYPE_CONVERT(bf16_t, bf16, pk_fp4_t, pk_fp4)
CK_TILE_SCALED_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp16_t, fp16)
CK_TILE_SCALED_TYPE_CONVERT(fp16_t, fp16, pk_fp4_t, pk_fp4)
#undef CK_TILE_SCALED_TYPE_CONVERT

} // namespace ck_tile
