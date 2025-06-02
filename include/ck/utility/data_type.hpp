// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/amd_ck_fp8.hpp"
#include "ck/utility/e8m0.hpp"
#include "ck/utility/statically_indexed_array.hpp"

/// Definitions from <cstdint>, <cmath> conflict with
/// /opt/rocm/include/hip/amd_detail/amd_hip_vector_types.h.

#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
#define CHAR_BIT 8
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

// scalar_type
template <typename TV>
struct scalar_type;

struct f4x2_pk_t
{
    static constexpr int packed_size = 2;

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

template <typename BitType, index_t pk_size>
struct f6_pk_t
{
    using element_type = uint32_t; // element storage fundamental type

    static constexpr index_t packed_size       = pk_size;
    static constexpr index_t num_bits_elem     = 6;
    static constexpr index_t num_bits_vec_elem = sizeof(element_type) * CHAR_BIT;
    static_assert((packed_size * num_bits_elem) % num_bits_vec_elem == 0,
                  "Packed elements must fit exactly into the element storage.");
    static constexpr index_t vector_size = (packed_size * num_bits_elem) / num_bits_vec_elem;

    using storage_type = StaticallyIndexedArray_v2<element_type, vector_size>;
    storage_type data; // packed data

    using type = f6_pk_t<BitType, packed_size>;

    __host__ __device__ constexpr f6_pk_t() : data{} {}
    __host__ __device__ constexpr f6_pk_t(storage_type init) : data{init} {}
    template <typename T, typename = enable_if_t<scalar_type<T>::vector_size == packed_size>>
    __host__ __device__ f6_pk_t(const T& v) : data{}
    {
        static_for<0, packed_size, 1>{}(
            [&](auto i) { pack(v[static_cast<index_t>(i)], static_cast<index_t>(i)); });
    }

    template <typename T>
    __host__ __device__ void pack(const T x, const index_t i)
    {
        static_assert(is_integral<T>::value || is_same_v<T, BitType>,
                      "T must be an integral type.");

        uint32_t bits        = static_cast<uint32_t>(x) & 0x3F;
        const int bit_pos    = i * num_bits_elem;
        const int arr_index  = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;
        uint32_t old_value   = data.data_[arr_index];

        // insert bits into the current 32-bit block
        old_value |= (bits << bit_offset);
        data.data_[arr_index] = old_value;

        // if it crosses into the next block, shift the remainder
        if(overhang > 0 && (arr_index + 1) < vector_size)
        {
            uint32_t next_value = data.data_[arr_index + 1];
            next_value |= (bits >> (num_bits_elem - overhang));
            data.data_[arr_index + 1] = next_value;
        }
    }

    __host__ __device__ static inline BitType unpack(const type& pk, const index_t i)
    {
        const int bit_pos    = i * num_bits_elem;
        const int arr_idx    = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;

        uint32_t bits = pk.data.data_[arr_idx] >> bit_offset;
        if(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (pk.data.data_[arr_idx + 1] & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<BitType>(bits & 0x3F);
    }

    __host__ __device__ inline BitType unpack(const index_t i) const { return unpack(*this, i); }
};

using f6x16_pk_t  = f6_pk_t<f6_t, 16>;
using f6x32_pk_t  = f6_pk_t<f6_t, 32>;
using bf6x16_pk_t = f6_pk_t<bf6_t, 16>;
using bf6x32_pk_t = f6_pk_t<bf6_t, 32>;

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
// native types: bool
template <typename T>
inline constexpr bool is_native_type()
{
    return is_same<T, double>::value || is_same<T, float>::value || is_same<T, half_t>::value ||
           is_same<T, bhalf_t>::value || is_same<T, int32_t>::value ||
           is_same<T, uint32_t>::value || is_same<T, int8_t>::value || is_same<T, uint8_t>::value ||
           is_same<T, f8_fnuz_t>::value || is_same<T, bf8_fnuz_t>::value || is_same<T, bool>::value;
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
struct scalar_type<f4x2_pk_t>
{
    using type                           = f4x2_pk_t::type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bool>
{
    using type                           = bool;
    static constexpr index_t vector_size = 1;
};

// Default behavior for types that do not need special handling
template <typename T>
struct packed_type
{
    using type                           = T;
    static constexpr index_t packed_size = 1; // number of packed elements
};

template <>
struct packed_type<int4_t>
{
    using type                           = pk_i4_t;
    static constexpr index_t packed_size = 2; // number of packed elements
};

template <>
struct packed_type<f4_t>
{
    using type                           = f4x2_pk_t;
    static constexpr index_t packed_size = 2; // number of packed elements
};

template <>
struct packed_type<f6_t>
{
    using type                           = f6x32_pk_t;
    static constexpr index_t packed_size = f6x32_pk_t::packed_size; // number of packed elements
};

template <>
struct packed_type<bf6_t>
{
    using type                           = bf6x32_pk_t;
    static constexpr index_t packed_size = bf6x32_pk_t::packed_size; // number of packed elements
};

template <typename T>
using packed_type_t = typename packed_type<T>::type;

// Check if the type has packed type specialization
template <typename T>
inline constexpr bool has_packed_type_v = !is_same_v<packed_type_t<T>, T>;

template <typename T>
struct element_type
{
    private:
    static constexpr auto get_element_type()
    {
        using U = remove_cvref_t<T>;
        if constexpr(is_same_v<U, pk_i4_t>)
            return int4_t{};
        else if constexpr(is_same_v<U, f4x2_pk_t>)
            return f4_t{};
        else if constexpr(is_same_v<U, f6x16_pk_t>)
            return f6_t{};
        else if constexpr(is_same_v<U, bf6x16_pk_t>)
            return bf6_t{};
        else if constexpr(is_same_v<U, f6x32_pk_t>)
            return f6_t{};
        else if constexpr(is_same_v<U, bf6x32_pk_t>)
            return bf6_t{};
        else
            return T{};
    }

    public:
    using type = decltype(get_element_type());
};
template <typename T>
using element_type_t = typename element_type<T>::type;

template <typename T>
inline constexpr bool is_packed_type_v =
    has_packed_type_v<element_type_t<T>>&& is_same_v<T, packed_type_t<element_type_t<T>>>;

template <typename T>
struct packed_size
{
    private:
    static constexpr auto get_packed_size()
    {
        using U = remove_cvref_t<T>;
        if constexpr(is_packed_type_v<U>)
            return Number<packed_type<element_type_t<U>>::packed_size>{};
        else
            return Number<packed_type<U>::packed_size>{};
    }

    public:
    using type                  = decltype(get_packed_size());
    static constexpr auto value = get_packed_size();
};

template <typename T>
using packed_size_t = typename packed_size<T>::type;

template <typename T>
inline constexpr index_t packed_size_v = packed_size<T>::value;

#if defined(_WIN32)
using int64_t = long long;
#else
using int64_t = long;
#endif

} // namespace ck
