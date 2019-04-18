#pragma once
#include "Sequence.hip.hpp"
#include "functional.hip.hpp"

template <class TData, index_t NSize>
struct Array
{
    using Type = Array<TData, NSize>;

    static constexpr index_t nSize = NSize;

    index_t mData[nSize];

    template <class... Xs>
    __host__ __device__ Array(Xs... xs) : mData{static_cast<TData>(xs)...}
    {
    }

    __host__ __device__ const TData& operator[](index_t i) const { return mData[i]; }

    __host__ __device__ TData& operator[](index_t i) { return mData[i]; }
};

template <class TData, index_t NSize, index_t... IRs>
__host__ __device__ auto reorder_array_given_new2old(const Array<TData, NSize>& old_array,
                                                     Sequence<IRs...> new2old)
{
    Array<TData, NSize> new_array;

    static_assert(NSize == sizeof...(IRs), "NSize not consistent");

    static_for<0, NSize, 1>{}([&](auto IDim) {
        constexpr index_t idim = IDim.Get();
        new_array[idim]        = old_array[new2old.Get(IDim)];
    });

    return new_array;
}

template <class TData, index_t NSize, index_t... IRs>
__host__ __device__ auto reorder_array_given_old2new(const Array<TData, NSize>& old_array,
                                                     Sequence<IRs...> old2new)
{
    Array<TData, NSize> new_array;

    static_assert(NSize == sizeof...(IRs), "NSize not consistent");

    static_for<0, NSize, 1>{}([&](auto IDim) {
        constexpr index_t idim       = IDim.Get();
        new_array[old2new.Get(IDim)] = old_array[idim];
    });

    return new_array;
}