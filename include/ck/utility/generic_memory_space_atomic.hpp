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

template <>
__device__ int8_t atomic_add<int8_t>(int8_t* p_dst, const int8_t& x)
{
    // Get the address of the 32-bit word containing the byte
    size_t byte_offset = reinterpret_cast<size_t>(p_dst) & 0x3;
    unsigned int* aligned_addr =
        reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(p_dst) & ~0x3);

    // Calculate bit position within the 32-bit word
    unsigned int byte_shift = byte_offset * 8;
    unsigned int byte_mask  = 0xFF << byte_shift;

    unsigned int old_word = *aligned_addr;
    unsigned int assumed_word;

    do
    {
        assumed_word = old_word;

        // Extract the byte, add to it, then put it back
        int8_t old_byte = static_cast<int8_t>((old_word >> byte_shift) & 0xFF);
        int8_t new_byte = old_byte + x;

        // Update just the byte in the 32-bit word
        unsigned int new_word =
            (old_word & ~byte_mask) | ((static_cast<unsigned int>(new_byte) & 0xFF) << byte_shift);

        old_word = atomicCAS(aligned_addr, assumed_word, new_word);

    } while(assumed_word != old_word);

    return static_cast<int8_t>((old_word >> byte_shift) & 0xFF);
}

template <>
__device__ int8x2_t atomic_add<int8x2_t>(int8x2_t* p_dst, const int8x2_t& x)
{
    constexpr auto I0 = Number<0>{};
    const vector_type<int8_t, 2> vx{x};
    vector_type<int8_t, 2> vy{0};

    static_for<0, 2, 1>{}([&](auto i) {
        vy.template AsType<int8_t>()(i) = atomic_add(c_style_pointer_cast<int8_t*>(p_dst) + i.value,
                                                     vx.template AsType<int8_t>()[i]);
    });

    return vy.template AsType<int8x2_t>()[I0];
}

template <>
__device__ int8x4_t atomic_add<int8x4_t>(int8x4_t* p_dst, const int8x4_t& x)
{
    constexpr auto I0 = Number<0>{};
    const vector_type<int8_t, 4> vx{x};
    vector_type<int8_t, 4> vy{0};

    static_for<0, 4, 1>{}([&](auto i) {
        vy.template AsType<int8_t>()(i) = atomic_add(c_style_pointer_cast<int8_t*>(p_dst) + i.value,
                                                     vx.template AsType<int8_t>()[i]);
    });

    return vy.template AsType<int8x4_t>()[I0];
}

template <>
__device__ int8x8_t atomic_add<int8x8_t>(int8x8_t* p_dst, const int8x8_t& x)
{
    constexpr auto I0 = Number<0>{};
    const vector_type<int8_t, 8> vx{x};
    vector_type<int8_t, 8> vy{0};

    static_for<0, 8, 1>{}([&](auto i) {
        vy.template AsType<int8_t>()(i) = atomic_add(c_style_pointer_cast<int8_t*>(p_dst) + i.value,
                                                     vx.template AsType<int8_t>()[i]);
    });

    return vy.template AsType<int8x8_t>()[I0];
}

template <>
__device__ int8x16_t atomic_add<int8x16_t>(int8x16_t* p_dst, const int8x16_t& x)
{
    constexpr auto I0 = Number<0>{};
    const vector_type<int8_t, 16> vx{x};
    vector_type<int8_t, 16> vy{0};

    static_for<0, 16, 1>{}([&](auto i) {
        vy.template AsType<int8_t>()(i) = atomic_add(c_style_pointer_cast<int8_t*>(p_dst) + i.value,
                                                     vx.template AsType<int8_t>()[i]);
    });

    return vy.template AsType<int8x16_t>()[I0];
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
