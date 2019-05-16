#pragma once
#include "vector_type.hip.hpp"
#include "constant_integral.hip.hpp"
#include "Sequence.hip.hpp"
#include "Array.hip.hpp"
#include "functional.hip.hpp"
#include "functional2.hip.hpp"

#if USE_AMD_INLINE_ASM
#include "amd_inline_asm.hip.hpp"
#endif

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

template <class T1, class T2>
struct is_same
{
    static const bool value = false;
};

template <class T>
struct is_same<T, T>
{
    static const bool value = true;
};

namespace mod_conv { // namespace mod_conv
template <class T, T s>
struct scales
{
    __host__ __device__ constexpr T operator()(T a) const { return s * a; }
};

template <class T>
struct integer_divide_ceiler
{
    __host__ __device__ constexpr T operator()(T a, T b) const
    {
        static_assert(is_same<T, index_t>::value || is_same<T, int>::value, "wrong type");

        return (a + b - 1) / b;
    }
};

template <class T>
__host__ __device__ constexpr T integer_divide_ceil(T a, T b)
{
    static_assert(is_same<T, index_t>::value || is_same<T, int>::value, "wrong type");

    return (a + b - 1) / b;
}

template <class T>
__host__ __device__ constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <class T, class... Ts>
__host__ __device__ constexpr T max(T x, Ts... xs)
{
    static_assert(sizeof...(xs) > 0, "not enough argument");

    auto y = max(xs...);

    static_assert(is_same<decltype(y), T>::value, "not the same type");

    return x > y ? x : y;
}

template <class T>
__host__ __device__ constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <class T, class... Ts>
__host__ __device__ constexpr T min(T x, Ts... xs)
{
    static_assert(sizeof...(xs) > 0, "not enough argument");

    auto y = min(xs...);

    static_assert(is_same<decltype(y), T>::value, "not the same type");

    return x < y ? x : y;
}
} // namespace mod_conv
