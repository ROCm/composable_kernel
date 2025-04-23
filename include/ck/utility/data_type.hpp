// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/amd_ck_fp8.hpp"
#include "ck/utility/e8m0.hpp"
#include "ck/utility/statically_indexed_array.hpp"

/// Definitions from <cstdint>, <cmath> conflict with
/// /opt/rocm/include/hip/amd_detail/amd_hip_vector_types.h.

#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
using int8_t   = signed char;
using uint8_t  = unsigned char;
using int16_t  = signed short;
using uint16_t = unsigned short;
using float_t  = float;
#endif // __HIPCC_RTC__

namespace ck {
#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
using byte = unsigned char;
#else
using std::byte;
#endif

using bhalf_t = ushort;
using half_t  = _Float16;
using int4_t  = _BitInt(4);
using f4_t    = unsigned _BitInt(4);
using f6_t    = _BitInt(6);          // e2m3 format
using bf6_t   = unsigned _BitInt(6); // e3m2 format

struct f4x2_pk_t
{
    using type = uint8_t;
    type data;
    __host__ __device__ f4x2_pk_t() : data{type{}} {}
    __host__ __device__ f4x2_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline type unpack(Number<I>) const
    {
        static_assert(I < 2, "Index is out of range.");
        if constexpr(I == 0)
            return (data >> 4);
        else
            return data & 0b00001111;
    }

    __host__ __device__ inline type pack(const type x0, const type x1)
    {
        return (x0 << 4) | (x1 & 0b00001111);
    }
};

struct f6x16_pk_t
{
    // store 16 elements of f6_t in an array of 3 uint32_t
    using element_type = uint32_t;
    using type         = StaticallyIndexedArray_v2<element_type, 3>;
    type data;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    f6x16_pk_t() : data{type{}} {}
    f6x16_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline f6_t unpack(Number<I>)
    {
        static_assert(I < 16, "Index out of range for 16 f6_t elements.");

        constexpr int num_bits_elem     = 6;
        constexpr int num_bits_vec_elem = 32;
        constexpr int vector_size       = 3;
        constexpr int bit_pos           = I * num_bits_elem;
        constexpr int arr_idx           = bit_pos / num_bits_vec_elem;
        constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
        uint32_t bits                   = data.At(Number<arr_idx>{}) >> bit_offset;
        constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;

        if constexpr(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (data.At(Number<arr_idx + 1>{}) & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<f6_t>(bits & 0x3F);
    }

    __host__ __device__ inline type pack(const test_vec_t& x)
    {
        type packed{};

        // for each of the 16 f6_t values, place its 6 bits in the correct position
        ck::static_for<0, 16, 1>{}([&](auto i) {
            uint32_t bits                   = static_cast<uint32_t>(x[static_cast<int>(i)]) & 0x3F;
            constexpr int num_bits_elem     = 6;
            constexpr int num_bits_vec_elem = 32;
            constexpr int vector_size       = 3;
            constexpr int bit_pos           = i * num_bits_elem;
            constexpr int arr_index         = bit_pos / num_bits_vec_elem;
            constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
            constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;
            uint32_t old_value              = packed.At(Number<arr_index>{});

            // insert bits into the current 32-bit block
            old_value |= (bits << bit_offset);
            packed.At(Number<arr_index>{}) = old_value;

            // if it crosses into the next block, shift the remainder
            if constexpr(overhang > 0 && (arr_index + 1) < vector_size)
            {
                uint32_t next_value = packed.At(Number<arr_index + 1>{});
                next_value |= (bits >> (num_bits_elem - overhang));
                packed.At(Number<arr_index + 1>{}) = next_value;
            }
        });

        return packed;
    }
};

struct f6x32_pk_t
{
    // store 32 elements of f6_t in an array of 6 uint32_t
    using element_type = uint32_t;
    using type         = StaticallyIndexedArray_v2<element_type, 6>;
    type data;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(32)));
    f6x32_pk_t() : data{type{}} {}
    f6x32_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline f6_t unpack(Number<I>)
    {
        static_assert(I < 32, "Index out of range for 32 f6_t elements.");

        constexpr int num_bits_elem     = 6;
        constexpr int num_bits_vec_elem = 32;
        constexpr int vector_size       = 6;
        constexpr int bit_pos           = I * num_bits_elem;
        constexpr int arr_idx           = bit_pos / num_bits_vec_elem;
        constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
        uint32_t bits                   = data.At(Number<arr_idx>{}) >> bit_offset;
        constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;

        if constexpr(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (data.At(Number<arr_idx + 1>{}) & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<f6_t>(bits & 0x3F);
    }

    __host__ __device__ inline type pack(const test_vec_t& x)
    {
        type packed{};

        // for each of the 32 f6_t values, place its 6 bits in the correct position
        ck::static_for<0, 32, 1>{}([&](auto i) {
            uint32_t bits                   = static_cast<uint32_t>(x[static_cast<int>(i)]) & 0x3F;
            constexpr int num_bits_elem     = 6;
            constexpr int num_bits_vec_elem = 32;
            constexpr int vector_size       = 6;
            constexpr int bit_pos           = i * num_bits_elem;
            constexpr int arr_index         = bit_pos / num_bits_vec_elem;
            constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
            constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;
            uint32_t old_value              = packed.At(Number<arr_index>{});

            // insert bits into the current 32-bit block
            old_value |= (bits << bit_offset);
            packed.At(Number<arr_index>{}) = old_value;

            // if it crosses into the next block, shift the remainder
            if constexpr(overhang > 0 && (arr_index + 1) < vector_size)
            {
                uint32_t next_value = packed.At(Number<arr_index + 1>{});
                next_value |= (bits >> (num_bits_elem - overhang));
                packed.At(Number<arr_index + 1>{}) = next_value;
            }
        });

        return packed;
    }
};

struct bf6x16_pk_t
{
    // store 16 elements of bf6_t in an array of 3 uint32_t
    using element_type = uint32_t;
    using type         = StaticallyIndexedArray_v2<element_type, 3>;
    type data;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    bf6x16_pk_t() : data{type{}} {}
    bf6x16_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline bf6_t unpack(Number<I>)
    {
        static_assert(I < 16, "Index out of range for 16 f6_t elements.");

        constexpr int num_bits_elem     = 6;
        constexpr int num_bits_vec_elem = 32;
        constexpr int vector_size       = 3;
        constexpr int bit_pos           = I * num_bits_elem;
        constexpr int arr_idx           = bit_pos / num_bits_vec_elem;
        constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
        uint32_t bits                   = data.At(Number<arr_idx>{}) >> bit_offset;
        constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;

        if constexpr(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (data.At(Number<arr_idx + 1>{}) & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<bf6_t>(bits & 0x3F);
    }

    __host__ __device__ inline type pack(const test_vec_t& x)
    {
        type packed{};

        // for each of the 16 bf6_t values, place its 6 bits in the correct position
        ck::static_for<0, 16, 1>{}([&](auto i) {
            uint32_t bits                   = static_cast<uint32_t>(x[static_cast<int>(i)]) & 0x3F;
            constexpr int num_bits_elem     = 6;
            constexpr int num_bits_vec_elem = 32;
            constexpr int vector_size       = 3;
            constexpr int bit_pos           = i * num_bits_elem;
            constexpr int arr_index         = bit_pos / num_bits_vec_elem;
            constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
            constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;
            uint32_t old_value              = packed.At(Number<arr_index>{});

            // insert bits into the current 32-bit block
            old_value |= (bits << bit_offset);
            packed.At(Number<arr_index>{}) = old_value;

            // if it crosses into the next block, shift the remainder
            if constexpr(overhang > 0 && (arr_index + 1) < vector_size)
            {
                uint32_t next_value = packed.At(Number<arr_index + 1>{});
                next_value |= (bits >> (num_bits_elem - overhang));
                packed.At(Number<arr_index + 1>{}) = next_value;
            }
        });

        return packed;
    }
};

struct bf6x32_pk_t
{
    // store 32 elements of bf6_t in an array of 6 uint32_t
    using element_type = uint32_t;
    using type         = StaticallyIndexedArray_v2<element_type, 6>;
    type data;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(32)));
    bf6x32_pk_t() : data{type{}} {}
    bf6x32_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline bf6_t unpack(Number<I>)
    {
        static_assert(I < 32, "Index out of range for 32 f6_t elements.");

        constexpr int num_bits_elem     = 6;
        constexpr int num_bits_vec_elem = 32;
        constexpr int vector_size       = 6;
        constexpr int bit_pos           = I * num_bits_elem;
        constexpr int arr_idx           = bit_pos / num_bits_vec_elem;
        constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
        uint32_t bits                   = data.At(Number<arr_idx>{}) >> bit_offset;
        constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;

        if constexpr(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (data.At(Number<arr_idx + 1>{}) & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<bf6_t>(bits & 0x3F);
    }

    __host__ __device__ inline type pack(const test_vec_t& x)
    {
        type packed{};

        // for each of the 32 bf6_t values, place its 6 bits in the correct position
        ck::static_for<0, 32, 1>{}([&](auto i) {
            uint32_t bits                   = static_cast<uint32_t>(x[static_cast<int>(i)]) & 0x3F;
            constexpr int num_bits_elem     = 6;
            constexpr int num_bits_vec_elem = 32;
            constexpr int vector_size       = 6;
            constexpr int bit_pos           = i * num_bits_elem;
            constexpr int arr_index         = bit_pos / num_bits_vec_elem;
            constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
            constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;
            uint32_t old_value              = packed.At(Number<arr_index>{});

            // insert bits into the current 32-bit block
            old_value |= (bits << bit_offset);
            packed.At(Number<arr_index>{}) = old_value;

            // if it crosses into the next block, shift the remainder
            if constexpr(overhang > 0 && (arr_index + 1) < vector_size)
            {
                uint32_t next_value = packed.At(Number<arr_index + 1>{});
                next_value |= (bits >> (num_bits_elem - overhang));
                packed.At(Number<arr_index + 1>{}) = next_value;
            }
        });

        return packed;
    }
};

// custom data type - pack int4 data
struct pk_i4_t
{
    using type = int8_t;
    type data;
    __host__ __device__ constexpr pk_i4_t() : data{type{}} {}
    __host__ __device__ constexpr pk_i4_t(type init) : data{init} {}
};

inline constexpr auto next_pow2(uint32_t x)
{
    // Precondition: x > 1.
    return x > 1u ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}

// native types: double, float, _Float16, ushort, int32_t, int8_t, uint8_t, f8_fnuz_t, bf8_fnuz_t,
// native types: bool, f4_t, f6_t, bf6_t
template <typename T>
inline constexpr bool is_native_type()
{
    return is_same<T, double>::value || is_same<T, float>::value || is_same<T, half_t>::value ||
           is_same<T, bhalf_t>::value || is_same<T, int32_t>::value || is_same<T, int8_t>::value ||
           is_same<T, uint8_t>::value || is_same<T, f8_fnuz_t>::value ||
           is_same<T, bf8_fnuz_t>::value || is_same<T, bool>::value || is_same<T, f4_t>::value ||
           is_same<T, f6_t>::value || is_same<T, bf6_t>::value;
}

// scalar_type
template <typename TV>
struct scalar_type;

// is_scalar_type
template <typename TV>
struct is_scalar_type
{
    static constexpr bool value = (scalar_type<remove_cvref_t<TV>>::vector_size == 1);
};

// has_same_scalar_type
template <typename X, typename Y>
using has_same_scalar_type = is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                     typename scalar_type<remove_cvref_t<Y>>::type>;

template <typename T, index_t N>
struct scalar_type<T __attribute__((ext_vector_type(N)))>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

//
template <>
struct scalar_type<double>
{
    using type                           = double;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<float>
{
    using type                           = float;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<half_t>
{
    using type                           = half_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bhalf_t>
{
    using type                           = bhalf_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int32_t>
{
    using type                           = int32_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int8_t>
{
    using type                           = int8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<uint8_t>
{
    using type                           = uint8_t;
    static constexpr index_t vector_size = 1;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct scalar_type<int4_t>
{
    using type                           = int4_t;
    static constexpr index_t vector_size = 1;
};
#endif

template <>
struct scalar_type<pk_i4_t>
{
    using type                           = pk_i4_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_fnuz_t>
{
    using type                           = f8_fnuz_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_fnuz_t>
{
    using type                           = bf8_fnuz_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_ocp_t>
{
    using type                           = f8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_ocp_t>
{
    using type                           = bf8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<e8m0_bexp_t>
{
    using type                           = e8m0_bexp_t::type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bool>
{
    using type                           = bool;
    static constexpr index_t vector_size = 1;
};

#if defined(_WIN32)
using int64_t = long long;
#else
using int64_t = long;
#endif

} // namespace ck
