#pragma once
#include "constant_integral.hip.hpp"
#include "functional.hip.hpp"

template <unsigned... Is>
struct Sequence
{
    using Type = Sequence<Is...>;

    static constexpr unsigned nDim = sizeof...(Is);

    const unsigned mData[nDim] = {Is...};

    template <unsigned I>
    __host__ __device__ constexpr unsigned Get(Number<I>) const
    {
        return mData[I];
    }

    // this is ugly, only for nDIm = 4
    template <unsigned I0, unsigned I1, unsigned I2, unsigned I3>
    __host__ __device__ constexpr auto ReorderByGetNewFromOld(Sequence<I0, I1, I2, I3>) const
    {
        static_assert(nDim == 4, "nDim != 4");

        constexpr auto old_sequence = Type{};

        constexpr unsigned NR0 = old_sequence.mData[I0];
        constexpr unsigned NR1 = old_sequence.mData[I1];
        constexpr unsigned NR2 = old_sequence.mData[I2];
        constexpr unsigned NR3 = old_sequence.mData[I3];

        return Sequence<NR0, NR1, NR2, NR3>{};
    }

    template <unsigned I0, unsigned I1, unsigned I2, unsigned I3>
    __host__ __device__ constexpr auto ReorderByPutOldToNew(Sequence<I0, I1, I2, I3>) const
    {
        // don't know how to implement this
        printf("Sequence::ReorderByPutOldToNew not implemented");
        assert(false);
    }

    template <unsigned I>
    __host__ __device__ constexpr auto PushBack(Number<I>) const
    {
        return Sequence<Is..., I>{};
    }

    __host__ __device__ constexpr auto PopBack() const;

    template <class F>
    __host__ __device__ constexpr auto Transform(F f) const
    {
        return Sequence<f(Is)...>{};
    }
};

template <unsigned... Is, unsigned I>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<Is..., I>)
{
    static_assert(sizeof...(Is) >= 1, "empty Sequence!");
    return Sequence<Is...>{};
}

template <class F, unsigned... Xs, unsigned... Ys>
__host__ __device__ constexpr auto sequence_sequence_op(Sequence<Xs...>, Sequence<Ys...>, F f)
{
    static_assert(Sequence<Xs...>::nDim == Sequence<Ys...>::nDim, "Dim not the same");

    return Sequence<f(Xs, Ys)...>{};
}

template <unsigned... Xs, unsigned... Ys>
__host__ __device__ constexpr auto sequence_sequence_add(Sequence<Xs...>, Sequence<Ys...>)
{
    struct add
    {
        __host__ __device__ constexpr unsigned operator()(unsigned x, unsigned y) const
        {
            return x + y;
        }
    };

    return sequence_sequence_op(Sequence<Xs...>{}, Sequence<Ys...>{}, add{});
}

template <unsigned... Is>
__host__ __device__ constexpr auto Sequence<Is...>::PopBack() const
{
    return sequence_pop_back(Type{});
}
