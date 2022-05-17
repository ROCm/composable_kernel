#pragma once

#include <immintrin.h>
#include "half.hpp"

namespace ck {

namespace cpu {

// vector_type
template <typename T, index_t N>
struct vector_type;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
#ifdef __clang__
template <typename T, index_t V, index_t N>
struct vector_type<T __attribute__((ext_vector_type(V))), N>;
#endif

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

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

template <>
struct vector_type<float, 1>
{
    using d1_t = float;
    // SSE
    using type = float;

    type data_;

    vector_type() : data_{0} {}

    // vector_type(float x) : data_{x} {}

    vector_type(type v) : data_{v} {}

    vector_type(const float* mem) : data_{*mem} {}

    template <typename X>
    constexpr const auto& AsType() const
    {
        static_assert(std::is_same<X, type>::value, "wrong!");

        return data_;
    }

    template <typename X>
    constexpr auto& AsType()
    {
        static_assert(std::is_same<X, type>::value, "wrong!");

        return data_;
    }

    constexpr void Load(const float* mem) { data_ = *mem; }

    constexpr void Store(float* mem) const { *mem = data_; }
};

template <>
struct vector_type<float, 4>
{
    using d1_t = float;
    // SSE
    using type = __m128;

    type data_;

    vector_type() : data_{_mm_setzero_ps()} {}

    vector_type(float x) : data_{_mm_set1_ps(x)} {}

    vector_type(type v) : data_{v} {}

    vector_type(const float* mem) : data_{_mm_loadu_ps(mem)} {}

    template <typename X>
    constexpr const auto& AsType() const
    {
        static_assert(std::is_same<X, type>::value, "wrong!");

        return data_;
    }

    template <typename X>
    constexpr auto& AsType()
    {
        static_assert(std::is_same<X, type>::value, "wrong!");

        return data_;
    }

    void Load(const float* mem) { data_ = _mm_loadu_ps(mem); }

    void Store(float* mem) const { _mm_storeu_ps(mem, data_); }
};

template <>
struct vector_type<float, 8>
{
    using d1_t = float;
    // SSE
    using type = __m256;

    type data_;

    vector_type() : data_{_mm256_setzero_ps()} {}

    vector_type(float x) : data_{_mm256_set1_ps(x)} {}

    vector_type(type v) : data_{v} {}

    vector_type(const float* mem) : data_{_mm256_loadu_ps(mem)} {}

    template <typename X>
    constexpr const auto& AsType() const
    {
        static_assert(std::is_same<X, type>::value, "wrong!");

        return data_;
    }

    template <typename X>
    constexpr auto& AsType()
    {
        static_assert(std::is_same<X, type>::value, "wrong!");

        return data_;
    }

    void Load(const float* mem) { data_ = _mm256_loadu_ps(mem); }

    void Store(float* mem) const { _mm256_storeu_ps(mem, data_); }
};

template <typename T>
struct to_vector_type
{
    using type = T;
};

template <>
struct to_vector_type<__m128>
{
    using type = vector_type<float, 4>;
};

template <>
struct to_vector_type<__m256>
{
    using type = vector_type<float, 8>;
};

template <typename Tv, typename Tp>
inline void load_vector(Tv& v, const Tp* mem)
{
    v = *reinterpret_cast<const Tv*>(mem);
}

template <>
inline void load_vector(__m128& v, const float* mem)
{
    v = _mm_loadu_ps(mem);
}

template <>
inline void load_vector(__m256& v, const float* mem)
{
    v = _mm256_loadu_ps(mem);
}

template <typename Tv, typename Tp>
inline void store_vector(const Tv& v, Tp* mem)
{
    *reinterpret_cast<Tv*>(mem) = v;
}

template <>
inline void store_vector(const __m128& v, float* mem)
{
    _mm_storeu_ps(mem, v);
}

template <>
inline void store_vector(const __m256& v, float* mem)
{
    _mm256_storeu_ps(mem, v);
}

template <typename Tv, typename Tx>
inline void set_vector(Tv& v, const Tx x)
{
    v = static_cast<const Tv>(x);
}

template <>
inline void set_vector(__m128& v, const float x)
{
    v = _mm_set1_ps(x);
}

template <>
inline void set_vector(__m256& v, const float x)
{
    v = _mm256_set1_ps(x);
}

template <typename Tv>
inline void clear_vector(Tv& v)
{
    v = static_cast<Tv>(0);
}

template <>
inline void clear_vector(__m128& v)
{
    v = _mm_setzero_ps();
}

template <>
inline void clear_vector(__m256& v)
{
    v = _mm256_setzero_ps();
}

using float4_t = typename vector_type<float, 4>::type;
using float8_t = typename vector_type<float, 8>::type;

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
struct scalar_type<vector_type<T, N>>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

template <>
struct scalar_type<float4_t>
{
    using type                           = float;
    static constexpr index_t vector_size = 4;
};

template <>
struct scalar_type<float8_t>
{
    using type                           = float;
    static constexpr index_t vector_size = 8;
};

//
template <>
struct scalar_type<float>
{
    using type                           = float;
    static constexpr index_t vector_size = 1;
};

} // namespace cpu
} // namespace ck
