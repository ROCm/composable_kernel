#pragma once
#include "constant_integral.hip.hpp"
#include "functional.hip.hpp"

template <index_t... Is>
struct Sequence
{
    using Type = Sequence;

    static constexpr index_t mSize = sizeof...(Is);

    const index_t mData[mSize + 1] = {
        Is..., 0}; // the last element is dummy, to prevent compiler complain on empty Sequence

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
        // TODO: don't know how to implement this
        printf("Sequence::ReorderGivenOld2New not implemented");
        assert(false);
    }

    __host__ __device__ constexpr auto Reverse() const;

    __host__ __device__ constexpr index_t Front() const { return mData[0]; }

    __host__ __device__ constexpr index_t Back() const { return mData[mSize - 1]; }

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

    template <index_t... Xs>
    __host__ __device__ constexpr auto Append(Sequence<Xs...>) const
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Ns>
    __host__ __device__ constexpr auto Extract(Number<Ns>...) const
    {
        return Sequence<Type{}.Get(Number<Ns>{})...>{};
    }

    template <index_t... Ns>
    __host__ __device__ constexpr auto Extract(Sequence<Ns...>) const
    {
        return Sequence<Type{}.Get(Number<Ns>{})...>{};
    }
};

template <class, class>
struct sequence_merge;

template <index_t... Xs, index_t... Ys>
struct sequence_merge<Sequence<Xs...>, Sequence<Ys...>>
{
    using SeqType = Sequence<Xs..., Ys...>;
};

template <index_t IBegin, index_t NSize, index_t Increment>
struct increasing_sequence_gen_impl
{
    static constexpr index_t NSizeLeft = NSize / 2;

    using SeqType = typename sequence_merge<
        typename increasing_sequence_gen_impl<IBegin, NSizeLeft, Increment>::SeqType,
        typename increasing_sequence_gen_impl<IBegin + NSizeLeft * Increment,
                                              NSize - NSizeLeft,
                                              Increment>::SeqType>::SeqType;
};

template <index_t IBegin, index_t Increment>
struct increasing_sequence_gen_impl<IBegin, 1, Increment>
{
    using SeqType = Sequence<IBegin>;
};

template <index_t IBegin, index_t Increment>
struct increasing_sequence_gen_impl<IBegin, 0, Increment>
{
    using SeqType = Sequence<>;
};

template <index_t IBegin, index_t IEnd, index_t Increment>
struct increasing_sequence_gen
{
    using SeqType =
        typename increasing_sequence_gen_impl<IBegin, IEnd - IBegin, Increment>::SeqType;
};

template <index_t IBegin, index_t IEnd, index_t Increment>
__host__ __device__ constexpr auto
    make_increasing_sequence(Number<IBegin>, Number<IEnd>, Number<Increment>)
{
    return typename increasing_sequence_gen<IBegin, IEnd, Increment>::SeqType{};
}

template <class, class>
struct sequence_reverse_inclusive_scan;

template <index_t I, index_t... Is, class Reduce>
struct sequence_reverse_inclusive_scan<Sequence<I, Is...>, Reduce>
{
    using old_scan = typename sequence_reverse_inclusive_scan<Sequence<Is...>, Reduce>::SeqType;

    static constexpr index_t new_reduce = Reduce{}(I, old_scan{}.Front());

    using SeqType = typename sequence_merge<Sequence<new_reduce>, old_scan>::SeqType;
};

template <index_t I, class Reduce>
struct sequence_reverse_inclusive_scan<Sequence<I>, Reduce>
{
    using SeqType = Sequence<I>;
};

template <class, class>
struct sequence_extract;

template <class Seq, index_t... Is>
struct sequence_extract<Seq, Sequence<Is...>>
{
    using SeqType = Sequence<Seq{}.Get(Number<Is>{})...>;
};

template <class Seq, index_t I>
struct sequence_split
{
    static constexpr index_t NSize = Seq{}.GetSize();

    using range0 = typename increasing_sequence_gen<0, I, 1>::SeqType;
    using range1 = typename increasing_sequence_gen<I, NSize, 1>::SeqType;

    using SeqType0 = typename sequence_extract<Seq, range0>::SeqType;
    using SeqType1 = typename sequence_extract<Seq, range1>::SeqType;
};

template <class Seq>
struct sequence_reverse
{
    static constexpr index_t NSize = Seq{}.GetSize();

    using seq_split = sequence_split<Seq, NSize / 2>;
    using SeqType   = typename sequence_merge<
        typename sequence_reverse<typename seq_split::SeqType1>::SeqType,
        typename sequence_reverse<typename seq_split::SeqType0>::SeqType>::SeqType;
};

template <index_t I>
struct sequence_reverse<Sequence<I>>
{
    using SeqType = Sequence<I>;
};

template <index_t I0, index_t I1>
struct sequence_reverse<Sequence<I0, I1>>
{
    using SeqType = Sequence<I1, I0>;
};

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs + Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator-(Sequence<Xs...> seq_x, Sequence<Ys...> seq_y)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    static_for<0, seq_x.GetSize(), 1>{}(
        [&](auto I) { static_assert(seq_x.Get(I) >= seq_y.Get(I), "wrong! going to undeflow"); });

    return Sequence<(Xs - Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator*(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs * Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator/(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs / Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator%(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs % Ys)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs + Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator-(Sequence<Xs...>, Number<Y>)
{
#if 0 // doesn't compile
    constexpr auto seq_x = Sequence<Xs...>{};

    static_for<0, sizeof...(Xs), 1>{}([&](auto Iter) {
        constexpr auto I = decltype(Iter){};
        static_assert(seq_x.Get(I) >= Y, "wrong! going to underflow");
    });
#endif

    return Sequence<(Xs - Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator*(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs * Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator/(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs / Y)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator%(Sequence<Xs...>, Number<Y>)
{
    return Sequence<(Xs % Y)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator+(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y + Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator-(Number<Y>, Sequence<Xs...>)
{
    constexpr auto seq_x = Sequence<Xs...>{};

    static_for<0, sizeof...(Xs), 1>{}([&](auto Iter) {
        constexpr auto I = decltype(Iter){};
        static_assert(seq_x.Get(I) <= Y, "wrong! going to underflow");
    });

    return Sequence<(Y - Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator*(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y * Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator/(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y / Xs)...>{};
}

template <index_t Y, index_t... Xs>
__host__ __device__ constexpr auto operator%(Number<Y>, Sequence<Xs...>)
{
    return Sequence<(Y % Xs)...>{};
}

template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_front(Sequence<I, Is...>)
{
    static_assert(sizeof...(Is) > 0, "empty Sequence!");
    return Sequence<Is...>{};
}

template <class Seq>
__host__ __device__ constexpr auto sequence_pop_back(Seq)
{
    static_assert(Seq{}.GetSize() > 0, "empty Sequence!");
    return sequence_pop_front(Seq{}.Reverse()).Reverse();
}

template <class F, index_t... Xs>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>)
{
    return Sequence<f(Xs)...>{};
}

template <class F, index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize, "Dim not the same");

    return Sequence<f(Xs, Ys)...>{};
}

template <class F, index_t... Xs, index_t... Ys, index_t... Zs>
__host__ __device__ constexpr auto
transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>, Sequence<Zs...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize &&
                      Sequence<Xs...>::mSize == Sequence<Zs...>::mSize,
                  "Dim not the same");

    return Sequence<f(Xs, Ys, Zs)...>{};
}

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
struct accumulate_on_sequence_impl
{
    template <class IDim>
    __host__ __device__ constexpr index_t operator()(IDim) const
    {
        return Seq{}.Get(IDim{});
    }
};

template <class Seq, class Reduce, index_t I>
__host__ __device__ constexpr index_t
    accumulate_on_sequence(Seq, Reduce, Number<I> /*initial_value*/)
{
    constexpr index_t a =
        static_const_reduce_n<Seq::mSize>{}(accumulate_on_sequence_impl<Seq>{}, Reduce{});
    return Reduce{}(a, I);
}

template <index_t... Is>
__host__ __device__ constexpr auto Sequence<Is...>::Reverse() const
{
    return typename sequence_reverse<Sequence<Is...>>::SeqType{};
}

template <class Seq, class Reduce>
__host__ __device__ constexpr auto reverse_inclusive_scan_sequence(Seq, Reduce)
{
    return typename sequence_reverse_inclusive_scan<Seq, Reduce>::SeqType{};
}

template <class Seq, class Reduce>
__host__ __device__ constexpr auto inclusive_scan_sequence(Seq, Reduce)
{
    return reverse_inclusive_scan_sequence(Seq{}.Reverse(), Reduce{}).Reverse();
}
