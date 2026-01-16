// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_STATICALLY_INDEXED_ARRAY_HPP
#define CK_STATICALLY_INDEXED_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"
#include "tuple.hpp"

namespace ck {

// StaticallyIndexedArray using simple C-array instead of template metaprogramming
// This avoids deep template instantiation while maintaining the same interface
template <typename T, index_t N>
struct StaticallyIndexedArray
{
    __host__ __device__ constexpr StaticallyIndexedArray() : data_{} {}

    // Single-element constructor - exclude containers with matching size (to prefer conversion
    // constructor)
    template <typename X>
        requires(N == 1 &&
                 // Allow if X is same type as T or doesn't have Size() method
                 (is_same<remove_cvref_t<X>, T>::value || !requires { remove_cvref_t<X>::Size(); }))
    __host__ __device__ constexpr StaticallyIndexedArray(X&& x)
        : data_{static_cast<T>(ck::forward<X>(x))}
    {
    }

    // Multi-element constructor
    template <typename... Xs>
        requires(sizeof...(Xs) == N && N > 1)
    __host__ __device__ constexpr StaticallyIndexedArray(Xs&&... xs)
        : data_{static_cast<T>(ck::forward<Xs>(xs))...}
    {
    }

    // Conversion constructor from any indexed container (Tuple, etc.)
    template <typename Container>
        requires(!is_same<remove_cvref_t<Container>, StaticallyIndexedArray>::value &&
                 requires { Container::Size(); } && Container::Size() == N)
    __host__ __device__ constexpr StaticallyIndexedArray(const Container& src)
        : StaticallyIndexedArray(
              make_from_container(src, typename arithmetic_sequence_gen<0, N, 1>::type{}))
    {
    }

    private:
    template <typename Container, index_t... Is>
    __host__ __device__ static constexpr StaticallyIndexedArray
    make_from_container(const Container& src, Sequence<Is...>)
    {
        return StaticallyIndexedArray{static_cast<T>(src[Number<Is>{}])...};
    }

    public:
    __host__ __device__ static constexpr index_t Size() { return N; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const T& At(Number<I>) const
    {
        static_assert(I < N, "wrong! out of range");
        return data_[I];
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr T& At(Number<I>)
    {
        static_assert(I < N, "wrong! out of range");
        return data_[I];
    }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const T& operator[](Number<I> i) const
    {
        return At(i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr T& operator()(Number<I> i)
    {
        return At(i);
    }

    template <typename U>
    __host__ __device__ constexpr auto operator=(const U& a)
    {
        static_assert(U::Size() == Size(), "wrong! size not the same");
        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });
        return *this;
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    T data_[N];
};

// Specialization for empty array
template <typename T>
struct StaticallyIndexedArray<T, 0>
{
    __host__ __device__ constexpr StaticallyIndexedArray() = default;

    __host__ __device__ static constexpr index_t Size() { return 0; }

    template <typename U>
    __host__ __device__ constexpr auto operator=(const U&)
    {
        return *this;
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }
};

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_statically_indexed_array(const X& x, const Xs&... xs)
{
    return StaticallyIndexedArray<X, sizeof...(Xs) + 1>{x, static_cast<X>(xs)...};
}

// make empty StaticallyIndexedArray
template <typename X>
__host__ __device__ constexpr auto make_statically_indexed_array()
{
    return StaticallyIndexedArray<X, 0>{};
}

template <typename T, index_t N>
struct StaticallyIndexedArray_v2
{
    __host__ __device__ constexpr StaticallyIndexedArray_v2() = default;

    __host__ __device__ static constexpr index_t Size() { return N; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I>) const
    {
        static_assert(I < N, "wrong! out of range");

        return data_[I];
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I>)
    {
        static_assert(I < N, "wrong! out of range");

        return data_[I];
    }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& operator()(Number<I> i)
    {
        return At(i);
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    T data_[N];
};

// Concepts for StaticallyIndexedArray arithmetic operators
template <typename T>
concept Scalar = ck::is_integral<T>::value || ck::is_floating_point<T>::value;

template <typename T>
concept IndexedContainer = !Scalar<T> && requires { T::Size(); };

// Arithmetic operators for StaticallyIndexedArray (to match Tuple operators)

// StaticallyIndexedArray += X
template <typename T, index_t N, IndexedContainer X>
__host__ __device__ constexpr auto operator+=(StaticallyIndexedArray<T, N>& y, const X& x)
{
    static_assert(X::Size() == N, "wrong! size not the same");
    static_for<0, N, 1>{}([&](auto i) { y(i) += x[i]; });
    return y;
}

// StaticallyIndexedArray -= X
template <typename T, index_t N, IndexedContainer X>
__host__ __device__ constexpr auto operator-=(StaticallyIndexedArray<T, N>& y, const X& x)
{
    static_assert(X::Size() == N, "wrong! size not the same");
    static_for<0, N, 1>{}([&](auto i) { y(i) -= x[i]; });
    return y;
}

// StaticallyIndexedArray + Y
template <typename T, index_t N, IndexedContainer Y>
__host__ __device__ constexpr auto operator+(const StaticallyIndexedArray<T, N>& x, const Y& y)
{
    static_assert(Y::Size() == N, "wrong! size not the same");
    StaticallyIndexedArray<T, N> r;
    static_for<0, N, 1>{}([&](auto i) { r(i) = x[i] + y[i]; });
    return r;
}

// StaticallyIndexedArray - Y
template <typename T, index_t N, IndexedContainer Y>
__host__ __device__ constexpr auto operator-(const StaticallyIndexedArray<T, N>& x, const Y& y)
{
    static_assert(Y::Size() == N, "wrong! size not the same");
    StaticallyIndexedArray<T, N> r;
    static_for<0, N, 1>{}([&](auto i) { r(i) = x[i] - y[i]; });
    return r;
}

// StaticallyIndexedArray * Y (element-wise)
template <typename T, index_t N, IndexedContainer Y>
__host__ __device__ constexpr auto operator*(const StaticallyIndexedArray<T, N>& x, const Y& y)
{
    static_assert(Y::Size() == N, "wrong! size not the same");
    StaticallyIndexedArray<T, N> r;
    static_for<0, N, 1>{}([&](auto i) { r(i) = x[i] * y[i]; });
    return r;
}

// scalar * StaticallyIndexedArray
template <typename T, index_t N, Scalar S>
__host__ __device__ constexpr auto operator*(S a, const StaticallyIndexedArray<T, N>& x)
{
    StaticallyIndexedArray<T, N> r;
    static_for<0, N, 1>{}([&](auto i) { r(i) = a * x[i]; });
    return r;
}

// StaticallyIndexedArray * scalar
template <typename T, index_t N, Scalar S>
__host__ __device__ constexpr auto operator*(const StaticallyIndexedArray<T, N>& x, S a)
{
    return a * x;
}

} // namespace ck
#endif
