// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "data_type.hpp"
#include "dtype_fp64.hpp"

namespace ck {

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_add explicit for
// each datatype.
template <typename X>
__device__ X atomic_add(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_add<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ uint32_t atomic_add<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float atomic_add<float>(float* p_dst, const float& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ unsigned short atomic_add<unsigned short>(unsigned short* p_dst, const unsigned short& x)
{
    // Use 32-bit aligned atomic operations
    uint32_t* aligned_addr = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(p_dst) & ~3U);

    const uint32_t byte_offset = reinterpret_cast<uintptr_t>(p_dst) & 3U;
    const uint32_t bit_shift   = byte_offset * 8;
    const uint32_t mask        = 0xFFFFU << bit_shift;

    uint32_t old_word, new_word;
    unsigned short old_val;

    do
    {
        old_word = *aligned_addr;
        old_val  = static_cast<unsigned short>((old_word >> bit_shift) & 0xFFFFU);

        uint32_t new_val = (static_cast<uint32_t>(old_val) + static_cast<uint32_t>(x)) & 0xFFFFU;
        new_word         = (old_word & ~mask) | (new_val << bit_shift);

    } while(atomicCAS(aligned_addr, old_word, new_word) != old_word);

    return old_val;
}

template <>
__device__ _Float16 atomic_add<_Float16>(_Float16* p_dst, const _Float16& x)
{
    // Use memcpy to avoid undefined behavior
    uint16_t* uint_dst = reinterpret_cast<uint16_t*>(p_dst);
    uint32_t* aligned_addr =
        reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(uint_dst) & ~3U);

    const uint32_t byte_offset = reinterpret_cast<uintptr_t>(uint_dst) & 3U;
    const uint32_t bit_shift   = byte_offset * 8;
    const uint32_t mask        = 0xFFFFU << bit_shift;

    uint32_t old_word, new_word;
    _Float16 old_val, new_val;

    do
    {
        old_word          = *aligned_addr;
        uint16_t old_bits = static_cast<uint16_t>((old_word >> bit_shift) & 0xFFFFU);

        // Use memcpy to avoid undefined behavior
        memcpy(&old_val, &old_bits, sizeof(uint16_t));
        new_val = old_val + x; // Proper FP16 addition

        uint16_t new_bits;
        memcpy(&new_bits, &new_val, sizeof(uint16_t));
        new_word = (old_word & ~mask) | (static_cast<uint32_t>(new_bits) << bit_shift);

    } while(atomicCAS(aligned_addr, old_word, new_word) != old_word);

    return old_val;
}

template <>
__device__ double atomic_add<double>(double* p_dst, const double& x)
{
    return atomicAdd(p_dst, x);
}

template <>
__device__ float2_t atomic_add<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

template <>
__device__ double2_t atomic_add<double2_t>(double2_t* p_dst, const double2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<double, 2> vx{x};
    vector_type<double, 2> vy{0};

    vy.template AsType<double>()(I0) =
        atomicAdd(c_style_pointer_cast<double*>(p_dst), vx.template AsType<double>()[I0]);
    vy.template AsType<double>()(I1) =
        atomicAdd(c_style_pointer_cast<double*>(p_dst) + 1, vx.template AsType<double>()[I1]);

    return vy.template AsType<double2_t>()[I0];
}

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_max explicit for
// each datatype.

template <typename X>
__device__ X atomic_max(X* p_dst, const X& x);

template <>
__device__ int32_t atomic_max<int32_t>(int32_t* p_dst, const int32_t& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ uint32_t atomic_max<uint32_t>(uint32_t* p_dst, const uint32_t& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ float atomic_max<float>(float* p_dst, const float& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ double atomic_max<double>(double* p_dst, const double& x)
{
    return atomicMax(p_dst, x);
}

template <>
__device__ float2_t atomic_max<float2_t>(float2_t* p_dst, const float2_t& x)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const vector_type<float, 2> vx{x};
    vector_type<float, 2> vy{0};

    vy.template AsType<float>()(I0) =
        atomicMax(c_style_pointer_cast<float*>(p_dst), vx.template AsType<float>()[I0]);
    vy.template AsType<float>()(I1) =
        atomicMax(c_style_pointer_cast<float*>(p_dst) + 1, vx.template AsType<float>()[I1]);

    return vy.template AsType<float2_t>()[I0];
}

} // namespace ck
