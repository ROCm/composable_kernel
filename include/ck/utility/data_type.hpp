// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/statically_indexed_array.hpp"

namespace ck {

using bhalf_t = ushort;
using half_t  = _Float16;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using int4_t = _BitInt(4);
#endif

struct f8_t
{
    uint8_t data;
    using type      = f8_t;
    using data_type = uint8_t;

    __host__ __device__ f8_t() = default;
    __host__ __device__ f8_t(uint8_t init);
};

struct bf8_t
{
    uint8_t data;
    using type      = bf8_t;
    using data_type = uint8_t;

    __host__ __device__ bf8_t() = default;
};

template <typename T>
inline __host__ __device__ constexpr auto is_native()
{
    return std::is_same<T, half_t>::value || std::is_same<T, float>::value ||
           std::is_same<T, double>::value || std::is_same<T, bhalf_t>::value ||
           std::is_same<T, int32_t>::value || std::is_same<T, int8_t>::value;
}

// vector_type
template <typename T, index_t N>
struct vector_type;

// // non_native_vector_type
// template <typename T, index_t N>
// struct non_native_vector_type;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<T __attribute__((ext_vector_type(V))), N>;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<vector_type<T, V>, N>;

// vector_type_maker
// This is the right way to handle "vector of vectors": making a bigger vector instead
template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<T __attribute__((ext_vector_type(N1))), N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
__host__ __device__ constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
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

template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
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

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct scalar_type<int4_t>
{
    using type                           = int4_t;
    static constexpr index_t vector_size = 1;
};
#endif

template <>
struct scalar_type<f8_t>
{
    using type                           = f8_t;
    static constexpr index_t vector_size = 1;
};

// utility function for non native vector type
inline constexpr auto next_pow2(uint32_t x)
{
    // Precondition: x > 1.
    return x > 1u ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}

//
template <typename T>
struct vector_type<T, 1>
{
    using d1_t = T;
    using type = d1_t;

    union
    {
        T d1_;
        StaticallyIndexedArray<T, 1> d1x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value, "wrong!");

        return data_.d1x1_;
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value, "wrong!");

        return data_.d1x1_;
    }
};

template <typename T>
struct vector_type<T, 2>
{
    using d1_t       = T;
    using ext_vect_t = conditional_t<is_native<T>(), T, uint32_t>;
    using d2_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(2))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(2))))>;

    using type = d2_t;

    union
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 4>
{
    using d1_t       = T;
    using ext_vect_t = conditional_t<is_native<T>(), T, uint32_t>;
    using d2_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(2))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(2))))>;
    using d4_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(4))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(4))))>;

    using type = d4_t;

    union
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 8>
{
    using d1_t       = T;
    using ext_vect_t = conditional_t<is_native<T>(), T, uint32_t>;
    using d2_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(2))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(2))))>;
    using d4_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(4))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(4))))>;
    using d8_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(8))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(8))))>;

    using type = d8_t;

    union
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 16>
{
    using d1_t       = T;
    using ext_vect_t = conditional_t<is_native<T>(), T, uint32_t>;
    using d2_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(2))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(2))))>;
    using d4_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(4))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(4))))>;
    using d8_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(8))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(8))))>;
    using d16_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(16))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(16))))>;

    using type = d16_t;

    union
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 32>
{
    using d1_t       = T;
    using ext_vect_t = conditional_t<is_native<T>(), T, uint32_t>;
    using d2_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(2))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(2))))>;
    using d4_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(4))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(4))))>;
    using d8_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(8))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(8))))>;
    using d16_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(16))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(16))))>;
    using d32_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(32))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(32))))>;

    using type = d32_t;

    union
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 64>
{
    using d1_t       = T;
    using ext_vect_t = conditional_t<is_native<T>(), T, uint32_t>;
    using d2_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(2))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(2))))>;
    using d4_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(4))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(4))))>;
    using d8_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(8))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(8))))>;
    using d16_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(16))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(16))))>;
    using d32_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(32))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(32))))>;
    using d64_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(64))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(64))))>;

    using type = d64_t;

    union
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 128>
{
    using d1_t       = T;
    using ext_vect_t = conditional_t<is_native<T>(), T, uint32_t>;
    using d2_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(2))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(2))))>;
    using d4_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(4))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(4))))>;
    using d8_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(8))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(8))))>;
    using d16_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(16))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(16))))>;
    using d32_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(32))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(32))))>;
    using d64_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(64))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(64))))>;
    using d128_t     = conditional_t<is_native<T>(),
                                 ext_vect_t __attribute__((ext_vector_type(128))),
                                 ext_vect_t __attribute__((ext_vector_type(next_pow2(128))))>;

    using type = d128_t;

    union
    {
        d128_t d128_;
        StaticallyIndexedArray<d1_t, 128> d1x128_;
        StaticallyIndexedArray<d2_t, 64> d2x64_;
        StaticallyIndexedArray<d4_t, 32> d4x32_;
        StaticallyIndexedArray<d8_t, 16> d8x16_;
        StaticallyIndexedArray<d16_t, 8> d16x8_;
        StaticallyIndexedArray<d32_t, 4> d32x4_;
        StaticallyIndexedArray<d64_t, 2> d64x2_;
        StaticallyIndexedArray<d128_t, 1> d128x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 256>
{
    using d1_t       = T;
    using ext_vect_t = conditional_t<is_native<T>(), T, uint32_t>;
    using d2_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(2))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(2))))>;
    using d4_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(4))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(4))))>;
    using d8_t       = conditional_t<is_native<T>(),
                               ext_vect_t __attribute__((ext_vector_type(8))),
                               ext_vect_t __attribute__((ext_vector_type(next_pow2(8))))>;
    using d16_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(16))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(16))))>;
    using d32_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(32))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(32))))>;
    using d64_t      = conditional_t<is_native<T>(),
                                ext_vect_t __attribute__((ext_vector_type(64))),
                                ext_vect_t __attribute__((ext_vector_type(next_pow2(64))))>;
    using d128_t     = conditional_t<is_native<T>(),
                                 ext_vect_t __attribute__((ext_vector_type(128))),
                                 ext_vect_t __attribute__((ext_vector_type(next_pow2(128))))>;
    using d256_t     = conditional_t<is_native<T>(),
                                 ext_vect_t __attribute__((ext_vector_type(256))),
                                 ext_vect_t __attribute__((ext_vector_type(next_pow2(256))))>;

    using type = d256_t;

    union
    {
        d256_t d256_;
        StaticallyIndexedArray<d1_t, 256> d1x256_;
        StaticallyIndexedArray<d2_t, 128> d2x128_;
        StaticallyIndexedArray<d4_t, 64> d4x64_;
        StaticallyIndexedArray<d8_t, 32> d8x32_;
        StaticallyIndexedArray<d16_t, 16> d16x16_;
        StaticallyIndexedArray<d32_t, 8> d32x8_;
        StaticallyIndexedArray<d64_t, 4> d64x4_;
        StaticallyIndexedArray<d128_t, 2> d128x2_;
        StaticallyIndexedArray<d256_t, 1> d256x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
    }
};

using int64_t = long;

// fp64
using double2_t = typename vector_type<double, 2>::type;
using double4_t = typename vector_type<double, 4>::type;

// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;
using half64_t = typename vector_type<half_t, 64>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;
using bhalf64_t = typename vector_type<bhalf_t, 64>::type;

// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

// f8
using f8x2_t  = typename vector_type<f8_t, 2>::type;
using f8x4_t  = typename vector_type<f8_t, 4>::type;
using f8x8_t  = typename vector_type<f8_t, 8>::type;
using f8x16_t = typename vector_type<f8_t, 16>::type;
using f8x32_t = typename vector_type<f8_t, 32>::type;
using f8x64_t = typename vector_type<f8_t, 64>::type;

template <typename T>
struct NumericLimits
{
    __host__ __device__ static constexpr T Min() { return std::numeric_limits<T>::min(); }

    __host__ __device__ static constexpr T Max() { return std::numeric_limits<T>::max(); }

    __host__ __device__ static constexpr T Lowest() { return std::numeric_limits<T>::lowest(); }

    __host__ __device__ static constexpr T QuietNaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

    __host__ __device__ static constexpr T Infinity() { return std::numeric_limits<T>::infinity(); }
};

template <>
struct NumericLimits<half_t>
{
    static constexpr unsigned short binary_min    = 0x0400;
    static constexpr unsigned short binary_max    = 0x7BFF;
    static constexpr unsigned short binary_lowest = 0xFBFF;
    static constexpr unsigned short binary_qnan   = 0x7FFF;

    __host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }

    __host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }

    __host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }

    __host__ __device__ static constexpr half_t QuietNaN() { return bit_cast<half_t>(binary_qnan); }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct NumericLimits<int4_t>
{
    __host__ __device__ static constexpr int4_t Min() { return int4_t(-8); }

    __host__ __device__ static constexpr int4_t Max() { return int4_t(7); }

    __host__ __device__ static constexpr int4_t Lowest() { return int4_t(-8); }
};
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4

template <>
struct NumericLimits<f8_t>
{
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    static constexpr uint8_t binary_max    = 0x77; // 0b01110111
    static constexpr uint8_t binary_lowest = 0xF7; // 0b11110111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000

    __host__ __device__ static f8_t Min()
    {
        f8_t x;
        x.data = binary_min;
        return x;
    }

    __host__ __device__ static f8_t Max()
    {
        f8_t x;
        x.data = binary_max;
        return x;
    }

    __host__ __device__ static f8_t Lowest()
    {
        f8_t x;
        x.data = binary_lowest;
        return x;
    }

    __host__ __device__ static f8_t QuietNaN()
    {
        f8_t x;
        x.data = binary_qnan;
        return x;
    }
};

} // namespace ck
