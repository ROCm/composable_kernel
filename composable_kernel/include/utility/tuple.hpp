#ifndef CK_TUPLE_HPP
#define CK_TUPLE_HPP

#include "integral_constant.hpp"

namespace ck {

template <class... Ts>
struct tuple : public std::tuple<Ts...>
{
    using type = tuple;

    __host__ __device__ static constexpr index_t GetSize() { return std::tuple_size(tuple{}); }

    template <index_t I>
    __host__ __device__ constexpr auto Get(Number<I>) const
    {
        return std::get<I>(*this);
    }

    template <index_t I>
    __host__ __device__ constexpr auto operator[](Number<I>) const
    {
        return Get(Number<I>{}) :
    }
};

// merge tuple
template <class... Tuples>
__host__ __device__ constexpr auto merge_tuple(Tuples&&... xs)
{
    return std::tuple_cat(xs...);
};

// generate sequence
template <index_t IBegin, index_t NRemain, class F>
struct tuple_gen_impl
{
    static constexpr index_t NRemainLeft  = NRemain / 2;
    static constexpr index_t NRemainRight = NRemain - NRemainLeft;
    static constexpr index_t IMiddle      = IBegin + NRemainLeft;

    using type =
        typename tuple_merge<typename tuple_gen_impl<IBegin, NRemainLeft, F>::type,
                             typename tuple_gen_impl<IMiddle, NRemainRight, F>::type>::type;
};

template <index_t I, class F>
struct tuple_gen_impl<I, 1, F>
{
    static constexpr auto x = F{}(Number<I>{});
    using type              = tuple<Is>;
};

template <index_t I, class F>
struct sequence_gen_impl<I, 0, F>
{
    using type = Sequence<>;
};

template <index_t NSize, class F>
struct sequence_gen
{
    using type = typename sequence_gen_impl<0, NSize, F>::type;
};

} // namespace ck
#endif
