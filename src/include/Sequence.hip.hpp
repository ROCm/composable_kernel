#pragma once
#include "constant_integral.hip.hpp"
#include "functional.hip.hpp"

struct EmptySequence
{
    template <class Seq>
    __host__ __device__ constexpr Seq Append(Seq) const
    {
        return {};
    }
};

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
        // TODO: don't know how to implement this
        printf("Sequence::ReorderGivenOld2New not implemented");
        assert(false);
    }

    __host__ __device__ constexpr auto Reverse() const
    {
        // not implemented
    }

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

    template <index_t Xs...>
    __host__ __device__ constexpr auto Append(Sequence<Xs...>) const
    {
        return Sequence<Is..., Xs...>{};
    }

    __host__ __device__ constexpr auto Append(EmptySequence) const { return Type{}; }

    template <index_t... Ns>
    __host__ __device__ constexpr auto Extract(Number<Ns>...) const
    {
        return Sequence<Type{}.Get(Number<Ns>)...>{};
    }

    template <index_t N>
    struct split_impl
    {
        template <class FirstSeq, class SecondSeq>
        __host__ __device__ constexpr auto operator()(FirstSeq, SecondSeq) const
        {
            constexpr new_first  = FirstSeq{}.PushBack(Number<Second{}.Front()>{});
            constexpr new_second = SecondSeq{}.PopFront();

            static_if<(N > 0)>{}([&](auto fwd) {
                return split_impl<N - 1>{}(new_first, fwd(new_second));
            }).else_([&](auto fwd) { return std::make_pair(new_first, fwd(new_second)); });
        }
    };

    // split one sequence to two sequnces: [0, I) and [I, nSize)
    // return type is std::pair
    template <index_t I>
    __host__ __device__ constexpr auto Split(Number<I>) const
    {
        static_assert(I <= nSize, "wrong! split position is too high!");

        static_if<(I == 0)>{}(
            [&](auto fwd) { return std::make_pair(EmptySequence<>{}, fwd(Type{})); });

        static_if<(I == nSize)>{}(
            [&](auto fwd) { return std::make_pair(Type<>{}, fwd(EmptySequence<>{})); });

        static_if<(I > 0 && I < nSize)>{}([&](auto fforwader) {
            constexpr auto first  = Sequence<Type{}.Front()> {}
            constexpr auto second = Type{}.PopFront();

            return split_impl<I - 1>{}(first, fwd(second));
        });
    }

    template <index_t I, index_t X>
    __host__ __device__ constexpr auto Modify(Number<I>, Number<X>) const
    {
        constexpr auto first_second = Split(Number<I>{});

        constexpr auto left  = first_second.first;
        constexpr auto right = first_second.second.PopFront();

        return left.PushBack(Number<X>{}).Append(right);
    }
};

template <index_t IBegin, index_t IEnd, index_t Increment>
__host__ __device__ auto make_increasing_sequence(Number<IBegin>, Number<IEnd>, Number<Increment>)
{
    static_assert(IBegin < IEnd, (IEnd - IBegin) % Increment == 0, "wrong!");

    // not implemented
}

template <index_t N, index_t X>
__host__ __device__ auto make_uniform_sequence(Number<N>, Number<X>);
{
    // not implemented
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Sequence<Ys...>) const
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs + Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator-(Sequence<Xs...> seq_x, Sequence<Ys...> seq_y) const
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    static_for<0, xs.GetSize(), 1>{}([&](auto I) { static_assert(seq_x.Get(I) >= seq_y.Get(I)); });

    return Sequence<(Xs - Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator*(Sequence<Xs...>, Sequence<Ys...>)const
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs * Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator/(Sequence<Xs...>, Sequence<Ys...>) const
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs / Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator%(Sequence<Xs...>, Sequence<Ys...>) const
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs % Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr auto operator%(Sequence<Xs...>, Sequence<Ys...>) const
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return Sequence<(Xs % Ys)...>{};
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator+(Sequence<Xs...>, Number<Y>) const
{
    return seq_x + make_uniform_sequence(Number<sizeof...(Xs)>, Number<Y>{});
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator-(Sequence<Xs...>, Number<Y>) const
{
    return seq_x - make_uniform_sequence(Number<sizeof...(Xs)>, Number<Y>{});
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator*(Sequence<Xs...>, Number<Y>)const
{
    return seq_x * make_uniform_sequence(Number<sizeof...(Xs)>, Number<Y>{});
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator/(Sequence<Xs...>, Number<Y>) const
{
    return seq_x / make_uniform_sequence(Number<sizeof...(Xs)>, Number<Y>{});
}

template <index_t... Xs, index_t Y>
__host__ __device__ constexpr auto operator%(Sequence<Xs...> seq_x, Number<Y> y) const
{
    return seq_x % make_uniform_sequence(Number<sizeof...(Xs)>, Number<Y>{});
}

template <index_t X, index_t... Ys>
__host__ __device__ constexpr auto operator+(Number<X>, Sequence<Ys...>) const
{
    return make_uniform_sequence(Number<sizeof...(Ys)>{}, Number<X>{}) + Sequence<Ys...>{};
}

template <index_t X, index_t... Ys>
__host__ __device__ constexpr auto operator-(Number<X>, Sequence<Ys...>) const
{
    return make_uniform_sequence(Number<sizeof...(Ys)>{}, Number<X>{}) - Sequence<Ys...>{};
}

template <index_t X, index_t... Ys>
__host__ __device__ constexpr auto operator*(Number<X>, Sequence<Ys...>)const
{
    return make_uniform_sequence(Number<sizeof...(Ys)>{}, Number<X>{}) * Sequence<Ys...>{};
}

template <index_t X, index_t... Ys>
__host__ __device__ constexpr auto operator/(Number<X>, Sequence<Ys...>) const
{
    return make_uniform_sequence(Number<sizeof...(Ys)>{}, Number<X>{}) / Sequence<Ys...>{};
}

template <index_t X, index_t... Ys>
__host__ __device__ constexpr auto operator%(Number<X>, Sequence<Ys...>) const
{
    return make_uniform_sequence(Number<sizeof...(Ys)>{}, Number<X>{}) % Sequence<Ys...>{};
}

template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_front(Sequence<I, Is...>)
{
    static_assert(sizeof...(Is) > 0, "empty Sequence!");
    return Sequence<Is...>{};
}

#if 0
// TODO: for some reason, compiler cannot instantiate this template
template <index_t I, index_t... Is>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<Is..., I>)
{
    static_assert(sizeof...(Is) > 0, "empty Sequence!");
    return Sequence<Is...>{};
}
#else
// TODO: delete these very ugly mess
template <index_t I0, index_t I1>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<I0, I1>)
{
    return Sequence<I0>{};
}

template <index_t I0, index_t I1, index_t I2>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<I0, I1, I2>)
{
    return Sequence<I0, I1>{};
}

template <index_t I0, index_t I1, index_t I2, index_t I3>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<I0, I1, I2, I3>)
{
    return Sequence<I0, I1, I2>{};
}

template <index_t I0, index_t I1, index_t I2, index_t I3, index_t I4>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<I0, I1, I2, I3, I4>)
{
    return Sequence<I0, I1, I2, I3>{};
}

template <index_t I0, index_t I1, index_t I2, index_t I3, index_t I4, index_t I5>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<I0, I1, I2, I3, I4, I5>)
{
    return Sequence<I0, I1, I2, I3, I4>{};
}

template <index_t I0, index_t I1, index_t I2, index_t I3, index_t I4, index_t I5, index_t I6>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<I0, I1, I2, I3, I4, I5, I6>)
{
    return Sequence<I0, I1, I2, I3, I4, I5>{};
}

template <index_t I0,
          index_t I1,
          index_t I2,
          index_t I3,
          index_t I4,
          index_t I5,
          index_t I6,
          index_t I7>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<I0, I1, I2, I3, I4, I5, I6, I7>)
{
    return Sequence<I0, I1, I2, I3, I4, I5, I6>{};
}

template <index_t I0,
          index_t I1,
          index_t I2,
          index_t I3,
          index_t I4,
          index_t I5,
          index_t I6,
          index_t I7,
          index_t I8>
__host__ __device__ constexpr auto sequence_pop_back(Sequence<I0, I1, I2, I3, I4, I5, I6, I7, I8>)
{
    return Sequence<I0, I1, I2, I3, I4, I5, I6, I7>{};
}

template <index_t I0,
          index_t I1,
          index_t I2,
          index_t I3,
          index_t I4,
          index_t I5,
          index_t I6,
          index_t I7,
          index_t I8,
          index_t I9>
__host__ __device__ constexpr auto
    sequence_pop_back(Sequence<I0, I1, I2, I3, I4, I5, I6, I7, I8, I9>)
{
    return Sequence<I0, I1, I2, I3, I4, I5, I6, I7, I8>{};
}
#endif

#if 1
// TODO: fix these mess
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
#else
// TODO:: these doesn't compile
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

template <index_t NRemain>
struct scan_sequence_impl
{
    template <class ScanedSeq, class RemainSeq, class Reduce>
    __host__ __device__ constexpr auto operator()(ScanedSeq, RemainSeq, Reduce) const
    {
        static_assert(RemainSeq{}.GetSize() == NRemain,
                      "wrong! RemainSeq and NRemain not consistent!");

        constexpr index_t a       = Reduce{}(ScanedSeq{}.Back(), RemainSeq{}.Front());
        constexpr auto scaned_seq = ScanedSeq{}.PushBack(Number<a>{});

        static_if<(NRemain > 1)>{}([&](auto fwd) {
            return scan_sequence_impl<NRemain - 1>{}(
                scaned_seq, RemainSeq{}.PopFront(), fwd(Reduce{}));
        }).else_([&](auto fwd) { return fwd(scaned_seq); });
    }
};

template <class Seq, class Reduce>
__host__ __device__ constexpr auto scan_sequence(Seq, Reduce)
{
    constexpr auto scaned_seq = Sequence<Seq{}.front()>{};
    constexpr auto remain_seq = Seq{}.PopFront();

    constexpr index_t remain_size = Seq::GetSize() - 1;

    return scan_sequence_impl<remain_size>{}(scaned_seq, remain_seq, Reduce{});
}

template <class Seq, class Reduce>
__host__ __device__ constexpr auto reverse_scan_sequence(Seq, Reduce)
{
    return scan_seqeunce(Seq{}.Reverse(), Reduce{}).Reverse();
}
