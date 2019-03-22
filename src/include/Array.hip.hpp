#pragma once

template <class TData, unsigned NSize>
struct Array
{
    using Type = Array<TData, NSize>;

    static constexpr unsigned nSize = NSize;

    unsigned mData[nSize];

    template <class... Xs>
    __host__ __device__ Array(Xs... xs) : mData{static_cast<TData>(xs)...}
    {
    }

    __host__ __device__ TData operator[](unsigned i) const { return mData[i]; }
};
