// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// we name this as ext_vector purposely, because clang ext_vector_type extention only accept literay
// basic type to construct a ext_vector_type you must be very careful using this, or will have lot
// of compiler errors e.g. struct A; using Ax2_t = A __attribute__((ext_vector_type(2)));  -> will
// have compiler error
namespace impl {
template <typename T_, index_t N_>
struct ext_vector
{
    static constexpr index_t N = N_;
    using value_type           = T_;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};
} // namespace impl

template <typename T, index_t N>
using ext_vector_t = typename impl::ext_vector<T, N>::type;

// by default, any type will result in a vector_size=1 with scalar_type=T traits.
// ... unless we have other vector_traits specialization
template <typename T>
struct vector_traits
{
    using scalar_type                    = remove_cvref_t<T>;
    static constexpr index_t vector_size = 1;
};

// specialization for ext_vector_type()
template <typename T, index_t N>
struct vector_traits<T __attribute__((ext_vector_type(N)))>
{
    using scalar_type                    = T;
    static constexpr index_t vector_size = N;
};

// below are some pre-defines of ext_vector_type
// attention! 2 vector type could be just the same type
// fp64
using fp64x2_t = double __attribute__((ext_vector_type(2)));
using fp64x4_t = double __attribute__((ext_vector_type(4)));

// fp32
using fp32x2_t  = float __attribute__((ext_vector_type(2)));
using fp32x4_t  = float __attribute__((ext_vector_type(4)));
using fp32x8_t  = float __attribute__((ext_vector_type(8)));
using fp32x16_t = float __attribute__((ext_vector_type(16)));
using fp32x32_t = float __attribute__((ext_vector_type(32)));
using fp32x64_t = float __attribute__((ext_vector_type(64)));

// fp16
using fp16x2_t  = _Float16 __attribute__((ext_vector_type(2)));
using fp16x4_t  = _Float16 __attribute__((ext_vector_type(4)));
using fp16x8_t  = _Float16 __attribute__((ext_vector_type(8)));
using fp16x16_t = _Float16 __attribute__((ext_vector_type(16)));
using fp16x32_t = _Float16 __attribute__((ext_vector_type(32)));
using fp16x64_t = _Float16 __attribute__((ext_vector_type(64)));

// bfp16
using bf16x2_t  = bf16_raw_t __attribute__((ext_vector_type(2)));
using bf16x4_t  = bf16_raw_t __attribute__((ext_vector_type(4)));
using bf16x8_t  = bf16_raw_t __attribute__((ext_vector_type(8)));
using bf16x16_t = bf16_raw_t __attribute__((ext_vector_type(16)));
using bf16x32_t = bf16_raw_t __attribute__((ext_vector_type(32)));
using bf16x64_t = bf16_raw_t __attribute__((ext_vector_type(64)));

// i32
using int32x2_t  = int32_t __attribute__((ext_vector_type(2)));
using int32x4_t  = int32_t __attribute__((ext_vector_type(4)));
using int32x8_t  = int32_t __attribute__((ext_vector_type(8)));
using int32x16_t = int32_t __attribute__((ext_vector_type(16)));
using int32x32_t = int32_t __attribute__((ext_vector_type(32)));
using int32x64_t = int32_t __attribute__((ext_vector_type(64)));

// i16
using int16x2_t  = int16_t __attribute__((ext_vector_type(2)));
using int16x4_t  = int16_t __attribute__((ext_vector_type(4)));
using int16x8_t  = int16_t __attribute__((ext_vector_type(8)));
using int16x16_t = int16_t __attribute__((ext_vector_type(16)));
using int16x32_t = int16_t __attribute__((ext_vector_type(32)));
using int16x64_t = int16_t __attribute__((ext_vector_type(64)));

// u16
using uint16x2_t  = uint16_t __attribute__((ext_vector_type(2)));
using uint16x4_t  = uint16_t __attribute__((ext_vector_type(4)));
using uint16x8_t  = uint16_t __attribute__((ext_vector_type(8)));
using uint16x16_t = uint16_t __attribute__((ext_vector_type(16)));
using uint16x32_t = uint16_t __attribute__((ext_vector_type(32)));
using uint16x64_t = uint16_t __attribute__((ext_vector_type(64)));

// i8
using int8x2_t  = int8_t __attribute((ext_vector_type(2)));
using int8x4_t  = int8_t __attribute((ext_vector_type(4)));
using int8x8_t  = int8_t __attribute((ext_vector_type(8)));
using int8x16_t = int8_t __attribute((ext_vector_type(16)));
using int8x32_t = int8_t __attribute((ext_vector_type(32)));
using int8x64_t = int8_t __attribute((ext_vector_type(64)));

// f8
using fp8x2_t  = fp8_raw_t __attribute((ext_vector_type(2)));
using fp8x4_t  = fp8_raw_t __attribute((ext_vector_type(4)));
using fp8x8_t  = fp8_raw_t __attribute((ext_vector_type(8)));
using fp8x16_t = fp8_raw_t __attribute((ext_vector_type(16)));
using fp8x32_t = fp8_raw_t __attribute((ext_vector_type(32)));
using fp8x64_t = fp8_raw_t __attribute((ext_vector_type(64)));

// bf8
using bf8x2_t  = bf8_raw_t __attribute((ext_vector_type(2)));
using bf8x4_t  = bf8_raw_t __attribute((ext_vector_type(4)));
using bf8x8_t  = bf8_raw_t __attribute((ext_vector_type(8)));
using bf8x16_t = bf8_raw_t __attribute((ext_vector_type(16)));
using bf8x32_t = bf8_raw_t __attribute((ext_vector_type(32)));
using bf8x64_t = bf8_raw_t __attribute((ext_vector_type(64)));

} // namespace ck_tile
