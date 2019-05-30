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
    __host__ __device__ constexpr Array(Xs... xs) : mData{static_cast<TData>(xs)...}
    {
    }

    __host__ __device__ constexpr index_t GetSize() const { return NSize; }

    __host__ __device__ const TData& operator[](index_t i) const { return mData[i]; }

    __host__ __device__ TData& operator[](index_t i) { return mData[i]; }

    __host__ __device__ auto PushBack(TData x) const
    {
        Array<TData, NSize + 1> new_array;

        static_for<0, NSize, 1>{}([&](auto I) {
            constexpr index_t i = I.Get();
            new_array[i]        = mData[i];
        });

        new_array[NSize] = x;

        return new_array;
    }
};

template <index_t... Is>
__host__ __device__ constexpr auto sequence2array(Sequence<Is...>)
{
    return Array<index_t, sizeof...(Is)>{Is...};
}

template <class TData, index_t NSize>
__host__ __device__ constexpr auto make_zero_array()
{
    Array<TData, NSize> a;

    static_for<0, NSize, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();
        a[i]                = static_cast<TData>(0);
    });

    return a;
}

template <class TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto reorder_array_given_new2old(const Array<TData, NSize>& old_array,
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
__host__ __device__ constexpr auto reorder_array_given_old2new(const Array<TData, NSize>& old_array,
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

template <class TData, index_t NSize, class ExtractSeq>
__host__ __device__ constexpr auto extract_array(const Array<TData, NSize>& old_array, ExtractSeq)
{
    Array<TData, ExtractSeq::GetSize()> new_array;

    constexpr index_t new_size = ExtractSeq::GetSize();

    static_assert(new_size <= NSize, "wrong! too many extract");

    static_for<0, new_size, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();
        new_array[i]        = old_array[ExtractSeq::Get(I)];
    });

    return new_array;
}

// Array = Array + Array
template <class TData, index_t NSize>
__host__ __device__ constexpr auto operator+(Array<TData, NSize> a, Array<TData, NSize> b)
{
    Array<TData, NSize> result;

    static_for<0, NSize, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();

        result[i] = a[i] + b[i];
    });

    return result;
}

// Array = Array - Array
template <class TData, index_t NSize>
__host__ __device__ constexpr auto operator-(Array<TData, NSize> a, Array<TData, NSize> b)
{
    Array<TData, NSize> result;

    static_for<0, NSize, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();

        result[i] = a[i] - b[i];
    });

    return result;
}

// Array = Array + Sequence
template <class TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator+(Array<TData, NSize> a, Sequence<Is...> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    static_for<0, NSize, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();

        result[i] = a[i] + b.Get(I);
    });

    return result;
}

// Array = Array - Sequence
template <class TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator-(Array<TData, NSize> a, Sequence<Is...> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    static_for<0, NSize, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();

        result[i] = a[i] - b.Get(I);
    });

    return result;
}

// Array = Array * Sequence
template <class TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator*(Array<TData, NSize> a, Sequence<Is...> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    static_for<0, NSize, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();

        result[i] = a[i] * b.Get(I);
    });

    return result;
}

// Array = Sequence - Array
template <class TData, index_t NSize, index_t... Is>
__host__ __device__ constexpr auto operator-(Sequence<Is...> a, Array<TData, NSize> b)
{
    static_assert(sizeof...(Is) == NSize, "wrong! size not the same");

    Array<TData, NSize> result;

    static_for<0, NSize, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();

        result[i] = a.Get(I) - b[i];
    });

    return result;
}

template <class TData, index_t NSize, class Reduce>
__host__ __device__ constexpr TData
accumulate_on_array(const Array<TData, NSize>& a, Reduce f, TData init)
{
    TData result = init;

    static_assert(NSize > 0, "wrong");

    static_for<0, NSize, 1>{}([&](auto I) {
        constexpr index_t i = I.Get();
        result              = f(result, a[i]);
    });

    return result;
}

template <class T, index_t NSize>
__host__ __device__ void print_Array(const char* s, Array<T, NSize> a)
{
    constexpr index_t nsize = a.GetSize();

    static_assert(nsize > 0 && nsize <= 10, "wrong!");

    static_if<nsize == 1>{}([&](auto) { printf("%s size %u, {%u}\n", s, nsize, a[0]); });

    static_if<nsize == 2>{}([&](auto) { printf("%s size %u, {%u %u}\n", s, nsize, a[0], a[1]); });

    static_if<nsize == 3>{}(
        [&](auto) { printf("%s size %u, {%u %u %u}\n", s, nsize, a[0], a[1], a[2]); });

    static_if<nsize == 4>{}(
        [&](auto) { printf("%s size %u, {%u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3]); });

    static_if<nsize == 5>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3], a[4]);
    });

    static_if<nsize == 6>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u}\n", s, nsize, a[0], a[1], a[2], a[3], a[4], a[5]);
    });

    static_if<nsize == 7>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u}\n",
               s,
               nsize,
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5],
               a[6]);
    });

    static_if<nsize == 8>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5],
               a[6],
               a[7]);
    });

    static_if<nsize == 9>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5],
               a[6],
               a[7],
               a[8]);
    });

    static_if<nsize == 10>{}([&](auto) {
        printf("%s size %u, {%u %u %u %u %u %u %u %u %u %u}\n",
               s,
               nsize,
               a[0],
               a[1],
               a[2],
               a[3],
               a[4],
               a[5],
               a[6],
               a[7],
               a[8],
               a[9]);
    });
}
