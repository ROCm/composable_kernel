#pragma once
#include "constant_integral.hip.hpp"
#include "functional.hip.hpp"

template <index_t... Is>
struct Sequence
{
    using Type = Sequence<Is...>;

    static constexpr index_t nDim = sizeof...(Is);

    const index_t mData[nDim] = {Is...};

    template <index_t I>
    __host__ __device__ constexpr index_t Get(Number<I>) const
    {
        return mData[I];
    }

    // this is ugly, only for nDIm = 4
    template <index_t I0, index_t I1, index_t I2, index_t I3>
    __host__ __device__ constexpr auto ReorderByGetNewFromOld(Sequence<I0, I1, I2, I3>) const
    {
        static_assert(nDim == 4, "nDim != 4");

        constexpr auto old_sequence = Type{};

        constexpr index_t NR0 = old_sequence.mData[I0];
        constexpr index_t NR1 = old_sequence.mData[I1];
        constexpr index_t NR2 = old_sequence.mData[I2];
        constexpr index_t NR3 = old_sequence.mData[I3];

        return Sequence<NR0, NR1, NR2, NR3>{};
    }

    template <index_t I0, index_t I1, index_t I2, index_t I3>
    __host__ __device__ constexpr auto ReorderByPutOldToNew(Sequence<I0, I1, I2, I3>) const
    {
        // don't know how to implement this
        printf("Sequence::ReorderByPutOldToNew not implemented");
        assert(false);
    }

    template <index_t I>
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

template <index_t... Is, index_t I>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<Is..., I>)
{
    static_assert(sizeof...(Is) >= 1, "empty Sequence!");
    return Sequence<Is...>{};
}

template <class F, index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto sequence_sequence_op(Sequence<Xs...>, Sequence<Ys...>, F f)
{
    static_assert(Sequence<Xs...>::nDim == Sequence<Ys...>::nDim, "Dim not the same");

    return Sequence<f(Xs, Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto sequence_sequence_add(Sequence<Xs...>, Sequence<Ys...>)
{
    struct add
    {
        __host__ __device__ constexpr index_t operator()(index_t x, index_t y) const
        {
            return x + y;
        }
    };

    return sequence_sequence_op(Sequence<Xs...>{}, Sequence<Ys...>{}, add{});
}

template <index_t... Is>
__host__ __device__ constexpr auto Sequence<Is...>::PopBack() const
{
    return sequence_pop_back(Type{});
}
