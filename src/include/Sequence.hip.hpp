#pragma once
#include "constant_integral.hip.hpp"
#include "functional.hip.hpp"

template <index_t... Is>
struct Sequence
{
    using Type = Sequence;

    static constexpr index_t mSize = sizeof...(Is);

    __host__ __device__ static constexpr index_t GetSize() { return mSize; }

    template <index_t I>
    __host__ __device__ static constexpr index_t Get(Number<I>)
    {
        static_assert(I < mSize, "wrong! I too large");

        // the last dummy element is to prevent compiler complain about empty Sequence
        const index_t mData[mSize + 1] = {Is..., 0};
        return mData[I];
    }

    __host__ __device__ index_t operator[](index_t i) const
    {
        const index_t mData[mSize + 1] = {Is..., 0};
        return mData[i];
    }

    template <index_t... IRs>
    __host__ __device__ static constexpr auto ReorderGivenNew2Old(Sequence<IRs...> /*new2old*/)
    {
#if 0 // require sequence_sort, which is not implemented yet
        static_assert(is_same<sequence_sort<Sequence<IRs...>>::SortedSeqType,
                              arithmetic_sequence_gen<0, mSize, 1>::SeqType>::value,
                      "wrong! invalid new2old map");
#endif

        return Sequence<Type{}.Get(Number<IRs>{})...>{};
    }

#if 0 // require sequence_sort, which is not implemented yet
    template <class MapOld2New>
    __host__ __device__ static constexpr auto ReorderGivenOld2New(MapOld2New /*old2new*/)
    {
        static_assert(is_same<sequence_sort<MapOld2New>::SortedSeqType,
                              arithmetic_sequence_gen<0, mSize, 1>::SeqType>::value,
                      "wrong! invalid old2new map");

        constexpr auto map_new2old = typename sequence_map_inverse<MapOld2New>::SeqMapType{};

        return ReorderGivenNew2Old(map_new2old);
    }
#endif

    __host__ __device__ static constexpr auto Reverse();

    __host__ __device__ static constexpr index_t Front()
    {
        const index_t mData[mSize + 1] = {Is..., 0};
        return mData[0];
    }

    __host__ __device__ static constexpr index_t Back()
    {
        const index_t mData[mSize + 1] = {Is..., 0};
        return mData[mSize - 1];
    }

    template <index_t I>
    __host__ __device__ static constexpr auto PushFront(Number<I>)
    {
        return Sequence<I, Is...>{};
    }

    template <index_t I>
    __host__ __device__ static constexpr auto PushBack(Number<I>)
    {
        return Sequence<Is..., I>{};
    }

    __host__ __device__ static constexpr auto PopFront();

    __host__ __device__ static constexpr auto PopBack();

    template <index_t... Xs>
    __host__ __device__ static constexpr auto Append(Sequence<Xs...>)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Number<Ns>...)
    {
        return Sequence<Type{}.Get(Number<Ns>{})...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Sequence<Ns...>)
    {
        return Sequence<Type{}.Get(Number<Ns>{})...>{};
    }

    template <index_t I, index_t X>
    __host__ __device__ static constexpr auto Modify(Number<I>, Number<X>);
};

template <class, class>
struct sequence_merge;

template <index_t... Xs, index_t... Ys>
struct sequence_merge<Sequence<Xs...>, Sequence<Ys...>>
{
    using SeqType = Sequence<Xs..., Ys...>;
};

template <index_t IBegin, index_t NSize, index_t Increment>
struct arithmetic_sequence_gen_impl
{
    static constexpr index_t NSizeLeft = NSize / 2;

    using SeqType = typename sequence_merge<
        typename arithmetic_sequence_gen_impl<IBegin, NSizeLeft, Increment>::SeqType,
        typename arithmetic_sequence_gen_impl<IBegin + NSizeLeft * Increment,
                                              NSize - NSizeLeft,
                                              Increment>::SeqType>::SeqType;
};

template <index_t IBegin, index_t Increment>
struct arithmetic_sequence_gen_impl<IBegin, 1, Increment>
{
    using SeqType = Sequence<IBegin>;
};

template <index_t IBegin, index_t Increment>
struct arithmetic_sequence_gen_impl<IBegin, 0, Increment>
{
    using SeqType = Sequence<>;
};

template <index_t IBegin, index_t IEnd, index_t Increment>
struct arithmetic_sequence_gen
{
    using SeqType =
        typename arithmetic_sequence_gen_impl<IBegin, IEnd - IBegin, Increment>::SeqType;
};

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

    using range0 = typename arithmetic_sequence_gen<0, I, 1>::SeqType;
    using range1 = typename arithmetic_sequence_gen<I, NSize, 1>::SeqType;

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

#if 0 // not fully implemented
template <class KeySeq0, class ValSeq0, class KeySeq1, class ValSeq1>
struct sequence_sort_merge_impl;

template <index_t Key0,
          index_t... Keys0,
          index_t Val0,
          index_t... Vals0,
          index_t Key1,
          index_t... Keys1,
          index_t Val0,
          index_t... Vals1>
struct sequence_sort_merge_impl<Sequence<Key0, Keys0...>,
                                Sequence<Val0, Vals0...>,
                                Sequence<Key1, Keys1...>,
                                Sequence<Val1, Vals1...>>
{
};

template <class>
struct sequence_sort;

template <index_t... Is>
struct sequence_sort<Sequence<Is...>>
{
    using OriginalSeqType        = Sequence<Is...>;
    using SortedSeqType          = xxxxx;
    using MapSorted2OriginalType = xxx;
};

template <class Seq, class IsValidSeqMap>
struct sequence_map_inverse_impl;

// impl for valid map, no impl for invalid map
template <index_t... Is>
struct sequence_map_inverse_impl<Sequence<Is...>, true>
{
    using SeqMapType = sequence_sort<Sequence<Is...>>::MapSorted2OriginalType;
};

template <class>
struct sequence_map_inverse;

template <class Is...>
struct sequence_map_inverse<Sequence<Is...>>
{
    // TODO: make sure the map to be inversed is valid: [0, sizeof...(Is))
    static constexpr bool is_valid_sequence_map =
        is_same<typename sequence_sort<Sequence<Is...>>::SortedSeqType,
                typename arithmetic_sequence_gen<0, sizeof...(Is), 1>::SeqType>::value;

    // make compiler fails, if is_valid_map != true
    using SeqMapType =
        typename sequence_map_inverse_impl<Sequence<Is...>, is_valid_map>::SeqMapType;
};
#endif

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
#if 0 // TODO: turn it on. Doesn't compile
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
    return Sequence<Is...>{};
}

template <class Seq>
__host__ __device__ constexpr auto sequence_pop_back(Seq)
{
    static_assert(Seq{}.GetSize() > 0, "wrong! cannot pop an empty Sequence!");
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
__host__ __device__ constexpr auto Sequence<Is...>::PopFront()
{
    return sequence_pop_front(Type{});
}

template <index_t... Is>
__host__ __device__ constexpr auto Sequence<Is...>::PopBack()
{
    return sequence_pop_back(Type{});
}

template <index_t... Is>
__host__ __device__ constexpr auto Sequence<Is...>::Reverse()
{
    return typename sequence_reverse<Sequence<Is...>>::SeqType{};
}

template <index_t... Is>
template <index_t I, index_t X>
__host__ __device__ constexpr auto Sequence<Is...>::Modify(Number<I>, Number<X>)
{
    static_assert(I < GetSize(), "wrong!");

    using seq_split          = sequence_split<Type, I>;
    constexpr auto seq_left  = typename seq_split::SeqType0{};
    constexpr auto seq_right = typename seq_split::SeqType1{}.PopFront();

    return seq_left.PushBack(Number<X>{}).Append(seq_right);
}
