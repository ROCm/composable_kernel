#pragma once

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
