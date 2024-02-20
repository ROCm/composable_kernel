#pragma once

#include "ck/host/utils.hpp"
#include "ck/host/seq.hpp"
namespace ck {
namespace host {

template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<Sequence<Is...>>
{
    template <class F>
    constexpr void operator()(F f) const
    {
        swallow{(f(Number<Is>{}), 0)...};
    }
};
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    constexpr static_for()
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "wrongs! should (Increment > 0 && NBegin <= NEnd) || (Increment < 0 && "
                      "NBegin >= NEnd)");
    }

    template <class F>
    constexpr void operator()(F f) const
    {
        static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(f);
    }
};

template <typename TData, index_t NSize>
struct Array
{
    using type      = Array;
    using data_type = TData;

    TData mData[NSize];

    static constexpr index_t Size() { return NSize; }

    constexpr const TData& At(index_t i) const { return mData[i]; }

    constexpr TData& At(index_t i) { return mData[i]; }

    constexpr const TData& operator[](index_t i) const { return At(i); }

    constexpr TData& operator()(index_t i) { return At(i); }

    template <typename T>
    constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }
};
template <index_t N>
using MultiIndex = Array<index_t, N>;

template <typename UpLengths,
          typename Coefficients,
          typename enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
struct Embed
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    UpLengths up_lengths_;
    Coefficients coefficients_;

    constexpr Embed() = default;

    constexpr Embed(const UpLengths& up_lengths, const Coefficients& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
    }

    static constexpr index_t GetNumOfLowerDimension() { return 1; }

    static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    constexpr void CalculateLowerIndex(LowIdx& idx_low, const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}([&idx_low, &idx_up, this](auto i) {
            idx_low(Number<0>{}) += idx_up[i] * this->coefficients_[i];
        });
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                          const UpIdxDiff& idx_diff_up,
                          LowIdx& idx_low,
                          const UpIdx&,
                          Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}(
            [&](auto i) { idx_diff_low(Number<0>{}) += idx_diff_up[i] * coefficients_[i]; });

        idx_low += idx_diff_low;
    }

    static constexpr bool IsLinearTransform() { return true; }

    static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex() { return true; }

    template <typename UpIdx>
    static constexpr bool IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<Coefficients>::value;
    }
};
template <typename UpLengths,
          typename Coefficients,
          typename enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
constexpr auto make_embed_transform(const UpLengths& up_lengths, const Coefficients& coefficients)
{
    return Embed<UpLengths, Coefficients>{up_lengths, coefficients};
}

template <index_t>
struct TupleElementKey
{
    constexpr TupleElementKey() = default;
};

template <typename Key, typename Data>
struct TupleElementKeyData
{
    using DataType = Data;

#if 0 // workaround compiler complaint about implicitly-deleted default constructor
     constexpr TupleElementKeyData() = default;
#else
    constexpr TupleElementKeyData() : mData{} {}
#endif

    template <
        typename T,
        typename enable_if<!std::is_same<ck::host::remove_cvref_t<T>, TupleElementKeyData>::value,
                           bool>::type = false>
    constexpr TupleElementKeyData(T&& v) : mData(std::forward<T>(v))
    {
    }

    DataType mData;
};

// for read access of tuple element
template <typename Key, typename Data>
constexpr const Data& get_tuple_element_data_reference(const TupleElementKeyData<Key, Data>& x)
{
    return static_cast<const Data&>(x.mData);
}

// for write access of tuple element
template <typename Key, typename Data>
constexpr Data& get_tuple_element_data_reference(TupleElementKeyData<Key, Data>& x)
{
    return x.mData;
}

// TODO: not sure the use of reference is correct
template <typename Key, typename Data>
constexpr Data&& get_tuple_element_data_reference(TupleElementKeyData<Key, Data>&& x)
{
    return static_cast<Data&&>(x.mData);
}

// for infering type of tuple element
template <typename Key, typename Data>
constexpr Data get_tuple_element_data(const TupleElementKeyData<Key, Data>& x)
{
    return std::forward(x.mData);
}

template <typename Indices, typename... Xs>
struct TupleImpl;

template <index_t... Is, typename... Xs>
struct TupleImpl<ck::host::Sequence<Is...>, Xs...> : TupleElementKeyData<TupleElementKey<Is>, Xs>...
{
    constexpr TupleImpl() = default;

    template <typename Y,
              typename enable_if<sizeof...(Is) == 1 && sizeof...(Xs) == 1 &&
                                     !std::is_same<ck::host::remove_cvref_t<Y>, TupleImpl>::value,
                                 bool>::type = false>
    constexpr TupleImpl(Y&& y) : TupleElementKeyData<TupleElementKey<Is>, Xs>(std::forward<Y>(y))...
    {
    }

    template <typename... Ys, typename enable_if<sizeof...(Ys) >= 2, bool>::type = false>
    constexpr TupleImpl(Ys&&... ys)
        : TupleElementKeyData<TupleElementKey<Is>, Xs>(std::forward<Ys>(ys))...
    {
        static_assert(sizeof...(Is) == sizeof...(Xs) && sizeof...(Is) == sizeof...(Ys),
                      "wrong! inconsistent size");
    }

    static constexpr index_t Size() { return sizeof...(Xs); }

    template <index_t I>
    constexpr const auto& GetElementDataByKey(TupleElementKey<I>) const
    {
        return get_tuple_element_data_reference<TupleElementKey<I>>(*this);
    }

    template <index_t I>
    constexpr auto& GetElementDataByKey(TupleElementKey<I>)
    {
        return get_tuple_element_data_reference<TupleElementKey<I>>(*this);
    }
};
template <typename... Xs>
struct Tuple : TupleImpl<typename arithmetic_sequence_gen<0, sizeof...(Xs), 1>::type, Xs...>
{
    using base = TupleImpl<typename arithmetic_sequence_gen<0, sizeof...(Xs), 1>::type, Xs...>;

    constexpr Tuple() = default;

    template <typename Y,
              typename enable_if<sizeof...(Xs) == 1 &&
                                     !std::is_same<ck::host::remove_cvref_t<Y>, Tuple>::value,
                                 bool>::type = false>
    constexpr Tuple(Y&& y) : base(std::forward<Y>(y))
    {
    }

    template <typename... Ys,
              typename enable_if<sizeof...(Ys) == sizeof...(Xs) && sizeof...(Ys) >= 2, bool>::type =
                  false>
    constexpr Tuple(Ys&&... ys) : base(std::forward<Ys>(ys)...)
    {
    }

    static constexpr index_t Size() { return sizeof...(Xs); }

    // read access
    template <index_t I>
    constexpr const auto& At(ck::host::Number<I>) const
    {
        static_assert(I < base::Size(), "wrong! out of range");
        return base::GetElementDataByKey(TupleElementKey<I>{});
    }

    // write access
    template <index_t I>
    constexpr auto& At(ck::host::Number<I>)
    {
        static_assert(I < base::Size(), "wrong! out of range");
        return base::GetElementDataByKey(TupleElementKey<I>{});
    }

    // read access
    template <index_t I>
    constexpr const auto& operator[](ck::host::Number<I> i) const
    {
        return At(i);
    }

    // write access
    template <index_t I>
    constexpr auto& operator()(ck::host::Number<I> i)
    {
        return At(i);
    }

    template <typename T>
    constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }

    static constexpr bool IsStaticBuffer() { return true; }

    static constexpr bool IsTuple() { return true; }
};

template <>
struct Tuple<>
{
    constexpr Tuple() = default;

    static constexpr index_t Size() { return 0; }

    template <typename T>
    constexpr auto operator=(const T&)
    {
        return *this;
    }

    static constexpr bool IsStaticBuffer() { return true; }
};

template <index_t I, typename TTuple>
struct tuple_element
{
    // type should keep the cv/ref qualifier of original tuple element
    using type = decltype(get_tuple_element_data<TupleElementKey<I>>(TTuple{}));
};

template <index_t I, typename TTuple>
using tuple_element_t = typename tuple_element<I, TTuple>::type;

template <typename... Xs>
constexpr auto make_tuple(Xs&&... xs)
{
    return Tuple<ck::host::remove_cvref_t<Xs>...>(std::forward<Xs>(xs)...);
}

// https://en.cppreference.com/w/cpp/utility/tuple/tie
template <typename... Args>
constexpr Tuple<Args&...> tie(Args&... args) noexcept
{
    return {args...};
}

template <typename Indices>
struct unpack_impl;

template <index_t... Is>
struct unpack_impl<Sequence<Is...>>
{
    template <typename F, typename X>
    constexpr auto operator()(F&& f, X&& x) const
    {
        return std::forward<F>(f)(std::forward<X>(x).At(Number<Is>{})...);
    }
};

template <typename F, typename X>
constexpr auto unpack(F&& f, X&& x)
{
    using X_ = remove_reference_t<X>;
    return unpack_impl<typename arithmetic_sequence_gen<0, X_::Size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x));
}

template <typename Seq0, typename Seq1>
struct unpack2_impl;

template <index_t... Is, index_t... Js>
struct unpack2_impl<Sequence<Is...>, Sequence<Js...>>
{
    template <typename F, typename X, typename Y>
    constexpr auto operator()(F&& f, X&& x, Y&& y) const
    {
        return std::forward<F>(f)(std::forward<X>(x).At(Number<Is>{})...,
                                  std::forward<Y>(y).At(Number<Js>{})...);
    }
};

template <typename F, typename X, typename Y>
constexpr auto unpack2(F&& f, X&& x, Y&& y)
{
    using X_ = remove_reference_t<X>;
    using Y_ = remove_reference_t<Y>;
    return unpack2_impl<typename arithmetic_sequence_gen<0, X_::Size(), 1>::type,
                        typename arithmetic_sequence_gen<0, Y_::Size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x), std::forward<Y>(y));
}

template <typename F, index_t N>
constexpr auto generate_tuple(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return make_tuple(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <typename F, typename X, index_t... Is>
constexpr auto transform_tuples_impl(F f, const X& x, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, index_t... Is>
constexpr auto transform_tuples_impl(F f, const X& x, const Y& y, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, typename Z, index_t... Is>
constexpr auto transform_tuples_impl(F f, const X& x, const Y& y, const Z& z, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}), z.At(Number<Is>{}))...);
}

template <typename F, typename X>
constexpr auto transform_tuples(F f, const X& x)
{
    return transform_tuples_impl(f, x, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y>
constexpr auto transform_tuples(F f, const X& x, const Y& y)
{
    return transform_tuples_impl(
        f, x, y, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y, typename Z>
constexpr auto transform_tuples(F f, const X& x, const Y& y, const Z& z)
{
    return transform_tuples_impl(
        f, x, y, z, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());

} // namespace host
} // namespace ck
