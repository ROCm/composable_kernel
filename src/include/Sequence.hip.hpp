#pragma once
#include "constant_integral.hip.hpp"
#include "functional.hip.hpp"

template <index_t... Is>
struct Sequence
{
    using Type = Sequence<Is...>;

    static constexpr index_t mSize = sizeof...(Is);

    const index_t mData[mSize] = {Is...};

    __host__ __device__ static constexpr index_t GetSize() { return mSize; }

    template <index_t I>
    __host__ __device__ constexpr index_t Get(Number<I>) const
    {
        return mData[I];
    }

    __host__ __device__ index_t operator[](index_t i) const { return mData[i]; }

    template <index_t... IRs>
    __host__ __device__ constexpr auto ReorderGivenNew2Old(Sequence<IRs...> /*new2old*/) const
    {
        static_assert(mSize == sizeof...(IRs), "mSize not consistent");

        constexpr auto old = Type{};

        return Sequence<old.Get(Number<IRs>{})...>{};
    }

    template <index_t... IRs>
    __host__ __device__ constexpr auto ReorderGivenOld2New(Sequence<IRs...> /*old2new*/) const
    {
        // don't know how to implement this
        printf("Sequence::ReorderGivenOld2New not implemented");
        assert(false);
    }

    template <index_t I>
    __host__ __device__ constexpr auto PushFront(Number<I>) const
    {
        return Sequence<I, Is...>{};
    }

    template <index_t I>
    __host__ __device__ constexpr auto PushBack(Number<I>) const
    {
        return Sequence<Is..., I>{};
    }

    __host__ __device__ constexpr auto PopFront() const;

    __host__ __device__ constexpr auto PopBack() const;

    template <class F>
    __host__ __device__ constexpr auto Transform(F f) const
    {
        return Sequence<f(Is)...>{};
    }
};

template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_front(Sequence<I, Is...>)
{
    static_assert(sizeof...(Is) > 0, "empty Sequence!");
    return Sequence<Is...>{};
}

template <index_t... Is, index_t I>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<Is..., I>)
{
    static_assert(sizeof...(Is) > 0, "empty Sequence!");
    return Sequence<Is...>{};
}

#if 1
// this is ugly, only for 2 sequences
template <class F, index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize, "Dim not the same");

    return Sequence<f(Xs, Ys)...>{};
}

// this is ugly, only for 3 sequences
template <class F, index_t... Xs, index_t... Ys, index_t... Zs>
__host__ __device__ constexpr auto
transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>, Sequence<Zs...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize &&
                      Sequence<Xs...>::mSize == Sequence<Zs...>::mSize,
                  "Dim not the same");

    return Sequence<f(Xs, Ys, Zs)...>{};
}
#else
template <index_t NRemain>
struct transform_sequences_impl
{
    template <class F, class Y, class... Xs>
    __host__ __device__ constexpr auto operator()(F f, Y y, Xs... xs) const
    {
        static_assert(NRemain > 1, "wrong! should have NRemain > 1");

        constexpr index_t N  = f(Xs{}.Get(Number<0>{})...);
        constexpr auto y_new = y.PushBack(Number<N>{});

        return transform_sequences_impl<NRemain - 1>{}(f, y_new, xs.PopFront()...);
    }
};

template <>
struct transform_sequences_impl<1>
{
    template <class F, class Y, class... Xs>
    __host__ __device__ constexpr auto operator()(F f, Y, Xs...) const
    {
        constexpr index_t N = f(Xs{}.Get(Number<0>{})...);
        return Y{}.PushBack(Number<N>{});
    }
};

template <class F, class X, class... Xs>
__host__ __device__ constexpr auto transform_sequences(F f, X x, Xs... xs)
{
    constexpr index_t nSize = X::GetSize();
    constexpr auto I0       = Number<0>{};

    constexpr auto y0 = Sequence<f(X{}.Get(I0), Xs{}.Get(I0)...)>{};

    return transform_sequences_impl<nSize - 1>{}(f, y0, x.PopFront(), xs.PopFront()...);
}
#endif

template <index_t... Is>
__host__ __device__ constexpr auto Sequence<Is...>::PopFront() const
{
    return sequence_pop_front(Type{});
}

template <index_t... Is>
__host__ __device__ constexpr auto Sequence<Is...>::PopBack() const
{
    return sequence_pop_back(Type{});
}

template <class Seq>
struct accumulate_on_sequence_f
{
    template <class IDim>
    __host__ __device__ constexpr index_t operator()(IDim) const
    {
        return Seq{}.Get(IDim{});
    }
};

template <class Seq, class Reduce, index_t I>
__host__ __device__ constexpr index_t accumulate_on_sequence(Seq, Reduce, Number<I>)
{
    constexpr index_t a =
        static_const_reduce_n<Seq::mSize>{}(accumulate_on_sequence_f<Seq>{}, Reduce{});
    return Reduce{}(a, I);
}
