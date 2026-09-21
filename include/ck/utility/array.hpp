// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_ARRAY_HPP
#define CK_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"
#include <type_traits>
#include <cassert>

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck {

template <typename TData, index_t NSize>
struct Array
{
    using type      = Array;
    using data_type = TData;

    TData mData[NSize];

    __host__ __device__ static constexpr index_t Size() { return NSize; }

    __host__ __device__ constexpr const TData& At(index_t i) const [[clang::lifetimebound]]
    {
        return mData[i];
    }

    __host__ __device__ constexpr TData& At(index_t i) [[clang::lifetimebound]] { return mData[i]; }

    __host__ __device__ constexpr const TData& operator[](index_t i) const [[clang::lifetimebound]]
    {
        return At(i);
    }

    __host__ __device__ constexpr TData& operator()(index_t i) [[clang::lifetimebound]]
    {
        return At(i);
    }

    template <typename... Args>
    __host__ constexpr auto Emplace(index_t i, Args&&... args)
        -> std::enable_if_t<std::is_nothrow_constructible_v<TData, Args&&...>>
    {
        assert(i >= 0 && i < NSize);
        mData[i].~TData();
        new(mData + i) TData(ck::forward<Args>(args)...);
    }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }
    __host__ __device__ constexpr const TData* begin() const [[clang::lifetimebound]]
    {
        return &mData[0];
    }
    __host__ __device__ constexpr const TData* end() const [[clang::lifetimebound]]
    {
        return &mData[NSize];
    }
    __host__ __device__ constexpr TData* begin() [[clang::lifetimebound]] { return &mData[0]; }
    __host__ __device__ constexpr TData* end() [[clang::lifetimebound]] { return &mData[NSize]; }
};

// empty Array
template <typename TData>
struct Array<TData, 0>
{
    using type      = Array;
    using data_type = TData;

    __host__ __device__ static constexpr index_t Size() { return 0; }
};

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_array(X&& x, Xs&&... xs)
{
    using data_type = remove_cvref_t<X>;
    return Array<data_type, sizeof...(Xs) + 1>{ck::forward<X>(x), ck::forward<Xs>(xs)...};
}

// make empty array
template <typename X>
__host__ __device__ constexpr auto make_array()
{
    return Array<X, 0>{};
}

} // namespace ck

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif

#endif
