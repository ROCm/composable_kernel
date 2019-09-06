#ifndef CK_TUPLE_HPP
#define CK_TUPLE_HPP

#include "integral_constant.hpp"
#include "Sequence.hpp"

namespace ck {

namespace detail {

template <index_t>
struct TupleElementKey
{
};

template <typename Key, typename Data>
struct TupleElement
{
    template <typename T>
    __host__ __device__ explicit constexpr TupleElement(T&& v) : mData(static_cast<T&&>(v))
    {
    }

    Data mData;
};

template <typename Key, typename Data>
__host__ __device__ constexpr const Data& get_tuple_element(const TupleElement<Key, Data>& x)
{
    return x.mData;
}

template <typename Key, typename Data>
__host__ __device__ constexpr Data& get_tuple_element(TupleElement<Key, Data>& x)
{
    return x.mData;
}

template <typename Key, typename Data>
__host__ __device__ constexpr Data&& get_tuple_element(TupleElement<Key, Data>&& x)
{
    return static_cast<Data&&>(x.mData);
}

template <typename Indices, typename... Xs>
struct TupleImpl;

template <index_t... Is, typename... Xs>
struct TupleImpl<Sequence<Is...>, Xs...> : TupleElement<TupleElementKey<Is>, Xs>...
{
    template <typename... Ys>
    __host__ __device__ explicit constexpr TupleImpl(Ys&&... ys)
        : TupleElement<TupleElementKey<Is>, Xs>(static_cast<Ys&&>(ys))...
    {
    }

    __host__ __device__ static constexpr index_t Size() { return sizeof...(Xs); }

    template <index_t I>
    __host__ __device__ constexpr const auto& GetElementByKey(TupleElementKey<I>) const
    {
        return get_tuple_element<TupleElementKey<I>>(*this);
    }

    template <index_t I>
    __host__ __device__ constexpr auto& GetElementByKey(TupleElementKey<I>)
    {
        return get_tuple_element<TupleElementKey<I>>(*this);
    }
};

} // namespace detail

template <typename... Xs>
struct Tuple : detail::TupleImpl<typename arithmetic_sequence_gen<0, sizeof...(Xs), 1>::type, Xs...>
{
    using base =
        detail::TupleImpl<typename arithmetic_sequence_gen<0, sizeof...(Xs), 1>::type, Xs...>;

    template <typename... Ys>
    __host__ __device__ explicit constexpr Tuple(Ys&&... ys) : base(static_cast<Ys&&>(ys)...)
    {
    }

    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I>) const
    {
        static_assert(I < base::Size(), "wrong! out of range");
        return GetElementByKey(detail::TupleElementKey<I>{});
    }

    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I>)
    {
        static_assert(I < base::Size(), "wrong! out of range");
        return GetElementByKey(detail::TupleElementKey<I>{});
    }
};

} // namespace ck
#endif
