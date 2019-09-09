#ifndef CK_SEQUENCE_HPP
#define CK_SEQUENCE_HPP

#include "integral_constant.hpp"
#include "functional.hpp"

namespace ck {

template <index_t, index_t, index_t>
struct static_for;

template <index_t...>
struct Sequence;

template <typename Seq, index_t I>
struct sequence_split;

template <typename>
struct sequence_reverse;

template <typename>
struct sequence_map_inverse;

template <typename>
struct is_valid_sequence_map;

template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_front(Sequence<I, Is...>);

template <typename Seq>
__host__ __device__ constexpr auto sequence_pop_back(Seq);

template <index_t... Is>
struct Sequence
{
    using Type      = Sequence;
    using data_type = index_t;

    static constexpr index_t mSize = sizeof...(Is);

    __host__ __device__ static constexpr auto Size() { return Number<mSize>{}; }

    __host__ __device__ static constexpr auto GetSize() { return Size(); }

    __host__ __device__ static constexpr index_t At(index_t I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const index_t mData[mSize + 1] = {Is..., 0};
        return mData[I];
    }

    template <index_t I>
    __host__ __device__ static constexpr auto At(Number<I>)
    {
        static_assert(I < mSize, "wrong! I too large");

        return Number<At(I)>{};
    }

    template <index_t I>
    __host__ __device__ static constexpr auto Get(Number<I>)
    {
        return At(Number<I>{});
    }

    template <typename I>
    __host__ __device__ constexpr auto operator[](I i) const
    {
        return At(i);
    }

    template <index_t... IRs>
    __host__ __device__ static constexpr auto ReorderGivenNew2Old(Sequence<IRs...> /*new2old*/)
    {
        static_assert(sizeof...(Is) == sizeof...(IRs),
                      "wrong! reorder map should have the same size as Sequence to be rerodered");

        static_assert(is_valid_sequence_map<Sequence<IRs...>>::value, "wrong! invalid reorder map");

        return Sequence<Type::At(Number<IRs>{})...>{};
    }

    // MapOld2New is Sequence<...>
    template <typename MapOld2New>
    __host__ __device__ static constexpr auto ReorderGivenOld2New(MapOld2New)
    {
        static_assert(MapOld2New::Size() == Size(),
                      "wrong! reorder map should have the same size as Sequence to be rerodered");

        static_assert(is_valid_sequence_map<MapOld2New>::value, "wrong! invalid reorder map");

        return ReorderGivenNew2Old(typename sequence_map_inverse<MapOld2New>::type{});
    }

    __host__ __device__ static constexpr auto Reverse()
    {
        return typename sequence_reverse<Type>::type{};
    }

    __host__ __device__ static constexpr auto Front()
    {
        static_assert(mSize > 0, "wrong!");
        return At(Number<0>{});
    }

    __host__ __device__ static constexpr auto Back()
    {
        static_assert(mSize > 0, "wrong!");
        return At(Number<mSize - 1>{});
    }

    __host__ __device__ static constexpr auto PopFront() { return sequence_pop_front(Type{}); }

    __host__ __device__ static constexpr auto PopBack() { return sequence_pop_back(Type{}); }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushFront(Sequence<Xs...>)
    {
        return Sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushFront(Number<Xs>...)
    {
        return Sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushBack(Sequence<Xs...>)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushBack(Number<Xs>...)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Number<Ns>...)
    {
        return Sequence<Type::At(Number<Ns>{})...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Sequence<Ns...>)
    {
        return Sequence<Type::At(Number<Ns>{})...>{};
    }

    template <index_t I, index_t X>
    __host__ __device__ static constexpr auto Modify(Number<I>, Number<X>)
    {
        static_assert(I < Size(), "wrong!");

        using seq_split          = sequence_split<Type, I>;
        constexpr auto seq_left  = typename seq_split::SeqType0{};
        constexpr auto seq_right = typename seq_split::SeqType1{}.PopFront();

        return seq_left.PushBack(Number<X>{}).PushBack(seq_right);
    }

    template <typename F>
    __host__ __device__ static constexpr auto Transform(F f)
    {
        return Sequence<f(Is)...>{};
    }
};

// merge sequence
template <typename Seq, typename... Seqs>
struct sequence_merge
{
    using type = typename sequence_merge<Seq, typename sequence_merge<Seqs...>::type>::type;
};

template <index_t... Xs, index_t... Ys>
struct sequence_merge<Sequence<Xs...>, Sequence<Ys...>>
{
    using type = Sequence<Xs..., Ys...>;
};

template <typename Seq>
struct sequence_merge<Seq>
{
    using type = Seq;
};

// generate sequence
template <index_t IBegin, index_t NRemain, typename F>
struct sequence_gen_impl
{
    static constexpr index_t NRemainLeft  = NRemain / 2;
    static constexpr index_t NRemainRight = NRemain - NRemainLeft;
    static constexpr index_t IMiddle      = IBegin + NRemainLeft;

    using type =
        typename sequence_merge<typename sequence_gen_impl<IBegin, NRemainLeft, F>::type,
                                typename sequence_gen_impl<IMiddle, NRemainRight, F>::type>::type;
};

template <index_t I, typename F>
struct sequence_gen_impl<I, 1, F>
{
    static constexpr index_t Is = F{}(Number<I>{});
    using type                  = Sequence<Is>;
};

template <index_t I, typename F>
struct sequence_gen_impl<I, 0, F>
{
    using type = Sequence<>;
};

template <index_t NSize, typename F>
struct sequence_gen
{
    using type = typename sequence_gen_impl<0, NSize, F>::type;
};

// arithmetic sequence
template <index_t IBegin, index_t IEnd, index_t Increment>
struct arithmetic_sequence_gen
{
    struct F
    {
        __host__ __device__ constexpr index_t operator()(index_t i) const
        {
            return i * Increment + IBegin;
        }
    };

    using type = typename sequence_gen<(IEnd - IBegin) / Increment, F>::type;
};

// uniform sequence
template <index_t NSize, index_t I>
struct uniform_sequence_gen
{
    struct F
    {
        __host__ __device__ constexpr index_t operator()(index_t) const { return I; }
    };

    using type = typename sequence_gen<NSize, F>::type;
};

// reverse inclusive scan (with init) sequence
template <typename, typename, index_t>
struct sequence_reverse_inclusive_scan;

template <index_t I, index_t... Is, typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<I, Is...>, Reduce, Init>
{
    using old_scan = typename sequence_reverse_inclusive_scan<Sequence<Is...>, Reduce, Init>::type;

    static constexpr index_t new_reduce = Reduce{}(I, old_scan{}.Front());

    using type = typename sequence_merge<Sequence<new_reduce>, old_scan>::type;
};

template <index_t I, typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<I>, Reduce, Init>
{
    using type = Sequence<Reduce{}(I, Init)>;
};

template <typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<Sequence<>, Reduce, Init>
{
    using type = Sequence<>;
};

// split sequence
template <typename Seq, index_t I>
struct sequence_split
{
    static constexpr index_t NSize = Seq{}.Size();

    using range0 = typename arithmetic_sequence_gen<0, I, 1>::type;
    using range1 = typename arithmetic_sequence_gen<I, NSize, 1>::type;

    using SeqType0 = decltype(Seq::Extract(range0{}));
    using SeqType1 = decltype(Seq::Extract(range1{}));
};

// reverse sequence
template <typename Seq>
struct sequence_reverse
{
    static constexpr index_t NSize = Seq{}.Size();

    using seq_split = sequence_split<Seq, NSize / 2>;
    using type      = typename sequence_merge<
        typename sequence_reverse<typename seq_split::SeqType1>::type,
        typename sequence_reverse<typename seq_split::SeqType0>::type>::type;
};

template <index_t I>
struct sequence_reverse<Sequence<I>>
{
    using type = Sequence<I>;
};

template <index_t I0, index_t I1>
struct sequence_reverse<Sequence<I0, I1>>
{
    using type = Sequence<I1, I0>;
};

template <typename Seq, typename Compare>
struct sequence_sort
{
    template <typename SeqLeft, typename SeqRight, typename MergedSeq, typename Comp>
    struct sorted_sequence_merge_impl
    {
        static constexpr bool pick_left     = SeqLeft::Front() < SeqRight::Front();
        static constexpr index_t next_value = pick_left ? SeqLeft::Front() : SeqRight::Front();

        using new_merged_seq = decltype(MergedSeq::PushBack(Number<next_value>{}));

        using new_left_seq =
            typename conditional<pick_left, decltype(SeqLeft::PopFront()), SeqLeft>::type;
        using new_right_seq =
            typename conditional<pick_left, SeqRight, decltype(SeqRight::PopFront())>::type;

        using type =
            typename sorted_sequence_merge_impl<new_left_seq, new_right_seq, new_merged_seq, Comp>::
                type;
    };

    template <typename SeqLeft, typename MergedSeq, typename Comp>
    struct sorted_sequence_merge_impl<SeqLeft, Sequence<>, MergedSeq, Comp>
    {
        using type = typename sequence_merge<MergedSeq, SeqLeft>::type;
    };

    template <typename SeqRight, typename MergedSeq, typename Comp>
    struct sorted_sequence_merge_impl<Sequence<>, SeqRight, MergedSeq, Comp>
    {
        using type = typename sequence_merge<MergedSeq, SeqRight>::type;
    };

    template <typename Seq0, typename Seq1, typename Comp>
    struct sorted_sequence_merge
    {
        using type = typename sorted_sequence_merge_impl<Seq0, Seq1, Sequence<>, Comp>::type;
    };

    using split          = sequence_split<Seq, Seq::Size() / 2>;
    using unsorted_left  = typename split::SeqType0;
    using unsorted_right = typename split::SeqType1;

    using sorted_left  = typename sequence_sort<unsorted_left, Compare>::type;
    using sorted_right = typename sequence_sort<unsorted_right, Compare>::type;

    using type = typename sorted_sequence_merge<sorted_left, sorted_right, Compare>::type;
};

template <index_t X, index_t Y, typename Compare>
struct sequence_sort<Sequence<X, Y>, Compare>
{
    static constexpr bool x_first = Compare{}(X, Y);

    using type = typename conditional<x_first, Sequence<X, Y>, Sequence<Y, X>>::type;
};

template <index_t X, typename Compare>
struct sequence_sort<Sequence<X>, Compare>
{
    using type = Sequence<X>;
};

template <typename Seq, typename Less, typename Equal>
struct sequence_unique_sort
{
    template <typename WorkInputSeq, typename WorkOutputSeq, typename Eq>
    struct sorted_sequence_uniquify_impl
    {
        static constexpr index_t new_value = WorkInputSeq::Front();
        using new_work_input_seq           = decltype(WorkInputSeq::PopFront());

        using new_working_output_seq =
            typename conditional<new_value == WorkOutputSeq::Back(),
                                 WorkOutputSeq,
                                 decltype(WorkOutputSeq::PopBack(Number<new_value>{}))>::type;
    };

    template <typename WorkInputSeq, typename Eq>
    struct sorted_sequence_uniquify_impl<WorkInputSeq, Sequence<>, Eq>
    {
        using type = WorkInputSeq;
    };

    template <typename SortedSeq, typename Eq>
    struct sorted_sequence_uniquify
    {
        using type = typename sorted_sequence_uniquify_impl<SortedSeq, Sequence<>, Eq>::type;
    };

    using sorted_seq = typename sequence_sort<Seq, Less>::type;

    using type = typename sorted_sequence_uniquify<sorted_seq, Equal>::type;
};

template <typename Seq>
struct is_valid_sequence_map
{
    // not implemented yet, always return true
    static constexpr integral_constant<bool, true> value = integral_constant<bool, true>{};

    // TODO: add proper check for is_valid, something like:
    // static constexpr bool value =
    //     is_same<typename arithmetic_sequence_gen<0, Seq::Size(), 1>::type,
    //             typename sequence_sort<Seq>::SortedSeqType>{};
};

template <typename X2Y, typename WorkingY2X, index_t XBegin, index_t XRemain>
struct sequence_map_inverse_impl
{
    private:
    static constexpr auto new_y2x = WorkingY2X::Modify(X2Y::At(Number<XBegin>{}), Number<XBegin>{});

    public:
    using type =
        typename sequence_map_inverse_impl<X2Y, decltype(new_y2x), XBegin + 1, XRemain - 1>::type;
};

template <typename X2Y, typename WorkingY2X, index_t XBegin>
struct sequence_map_inverse_impl<X2Y, WorkingY2X, XBegin, 0>
{
    using type = WorkingY2X;
};

template <typename X2Y>
struct sequence_map_inverse
{
    using type =
        typename sequence_map_inverse_impl<X2Y,
                                           typename uniform_sequence_gen<X2Y::Size(), 0>::type,
                                           0,
                                           X2Y::Size()>::type;
};

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs + Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator-(Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

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

template <typename Seq>
__host__ __device__ constexpr auto sequence_pop_back(Seq)
{
    static_assert(Seq::Size() > 0, "wrong! cannot pop an empty Sequence!");
    return sequence_pop_front(Seq::Reverse()).Reverse();
}

template <typename F, index_t... Xs>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>)
{
    return Sequence<f(Xs)...>{};
}

template <typename... Seqs>
__host__ __device__ constexpr auto merge_sequences(Seqs...)
{
    return typename sequence_merge<Seqs...>::type{};
}

template <typename F, index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize, "Dim not the same");

    return Sequence<f(Xs, Ys)...>{};
}

template <typename F, index_t... Xs, index_t... Ys, index_t... Zs>
__host__ __device__ constexpr auto
transform_sequences(F f, Sequence<Xs...>, Sequence<Ys...>, Sequence<Zs...>)
{
    static_assert(Sequence<Xs...>::mSize == Sequence<Ys...>::mSize &&
                      Sequence<Xs...>::mSize == Sequence<Zs...>::mSize,
                  "Dim not the same");

    return Sequence<f(Xs, Ys, Zs)...>{};
}

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr auto reverse_inclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return typename sequence_reverse_inclusive_scan<Seq, Reduce, Init>::type{};
}

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr auto inclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq{}.Reverse(), Reduce{}, Number<Init>{}).Reverse();
}

template <typename Seq, typename Reduce>
struct lambda_accumulate_on_sequence
{
    const Reduce& f;
    index_t& result;

    __host__ __device__ constexpr lambda_accumulate_on_sequence(const Reduce& f_, index_t& result_)
        : f(f_), result(result_)
    {
    }

    template <typename IDim>
    __host__ __device__ constexpr index_t operator()(IDim) const
    {
        return result = f(result, Seq::At(IDim{}));
    }
};

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr index_t
accumulate_on_sequence(Seq, Reduce f, Number<Init> /*initial_value*/)
{
    index_t result = Init;

    static_for<0, Seq::mSize, 1>{}(lambda_accumulate_on_sequence<Seq, Reduce>(f, result));

    return result;
}

} // namespace ck
#endif
