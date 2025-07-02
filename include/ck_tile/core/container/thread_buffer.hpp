// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/tuple.hpp"

namespace ck_tile {

#if CK_TILE_THREAD_BUFFER_DEFAULT == CK_TILE_THREAD_BUFFER_USE_TUPLE
template <typename T, index_t N>
using thread_buffer = tuple_array<T, N>;

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_thread_buffer(Ts&&... ts)
{
    return make_tuple(ts...);
}
#else

#if 0
template <typename T, index_t N>
using thread_buffer = array<T, N>;

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_thread_buffer(Ts&&... ts)
{
    return make_array(ts...);
}

#endif

// clang-format off
template<typename T_, index_t N_>
struct thread_buffer {
    using value_type = remove_cvref_t<T_>;
    static constexpr index_t N = N_;

    value_type data[N];

    // TODO: this ctor can't ignore
    CK_TILE_HOST_DEVICE constexpr thread_buffer() : data{} {}
    CK_TILE_HOST_DEVICE constexpr thread_buffer(const value_type & o) : data{o} {}

    CK_TILE_HOST_DEVICE static constexpr auto size() { return N; }
    CK_TILE_HOST_DEVICE auto & get() {return data; }
    CK_TILE_HOST_DEVICE const auto & get() const {return data; }
    CK_TILE_HOST_DEVICE auto & get(index_t i) {return data[i]; }
    CK_TILE_HOST_DEVICE const auto & get(index_t i) const {return data[i]; }
    CK_TILE_HOST_DEVICE constexpr const auto& operator[](index_t i) const { return get(i); }
    CK_TILE_HOST_DEVICE constexpr auto& operator[](index_t i)             { return get(i); }
    CK_TILE_HOST_DEVICE constexpr auto& operator()(index_t i)             { return get(i); }     // TODO: compatible
    CK_TILE_HOST_DEVICE constexpr auto& at(index_t i)                                   { return get(i); }
    CK_TILE_HOST_DEVICE constexpr const auto& at(index_t i) const                       { return get(i); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& at()                       { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& at() const           { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& at(number<I>)              { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& at(number<I>) const  { return get(I); }
    
    template <typename X_,
              typename std::enable_if<has_same_scalar_type<value_type, X_>::value, bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr auto _get_as() const
    {
        using X = remove_cvref_t<X_>;

        constexpr index_t kSPerX = vector_traits<X>::vector_size;
        static_assert(N % kSPerX == 0);

        union {
            thread_buffer<X_, N / kSPerX> data {};
            // tuple_array<value_type, kSPerX> sub_data;
            value_type sub_data[N];
        } vx;
        static_for<0, N, 1>{}(
            [&](auto j) { vx.sub_data[j] = data[j]; });
        return vx.data;
    }

    template <typename X_,
              index_t Is,
              typename std::enable_if<has_same_scalar_type<value_type, X_>::value, bool>::type = false>
    CK_TILE_HOST_DEVICE const constexpr remove_reference_t<X_> _get_as(number<Is> is) const
    {
        using X = remove_cvref_t<X_>;

        constexpr index_t kSPerX = vector_traits<X>::vector_size;

        union {
            X_ data {};
            tuple_array<value_type, kSPerX> sub_data;
        } vx;
        static_for<0, kSPerX, 1>{}(
            [&](auto j) { vx.sub_data(j) = operator[]((is * number<sizeof(X_)/sizeof(value_type)>{}) + j); });
        return vx.data;
    }

#if 0
    template <typename X_,
              index_t Is,
              typename std::enable_if<has_same_scalar_type<value_type, X_>::value, bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void _set_as(number<Is> is, X_ x)
    {
        using X = remove_cvref_t<X_>;

        constexpr index_t kSPerX = vector_traits<X>::vector_size;

        union {
            X_ data;
            tuple_array<value_type, kSPerX> sub_data;
        } vx {x};

        static_for<0, kSPerX, 1>{}(
           [&](auto j) { operator()((is * number<sizeof(X_)/sizeof(value_type)>{}) + j) = vx.sub_data[j]; });
    }
#endif


#define TB_COMMON_AS() \
            static_assert(sizeof(value_type) * N % sizeof(Tx) == 0); \
            constexpr int vx = sizeof(value_type) * N / sizeof(Tx)

    template<typename Tx>
    CK_TILE_HOST_DEVICE auto & get_as() {TB_COMMON_AS();
            return reinterpret_cast<thread_buffer<Tx, vx>&>(data);}
    template<typename Tx>
    CK_TILE_HOST_DEVICE constexpr auto get_as() const {TB_COMMON_AS();
            if constexpr(sizeof(value_type) <= 1 )
            return _get_as<Tx>();   // TODO: current compiler for 8bit data need use union to get data back, should fix in the future
            else
            return reinterpret_cast<const thread_buffer<Tx, vx>&>(data);}
    template<typename Tx, index_t I>
    CK_TILE_HOST_DEVICE auto & get_as(number<I>) {TB_COMMON_AS();
            return reinterpret_cast<thread_buffer<Tx, vx>&>(data).get(number<I>{});}
    template<typename Tx, index_t I>
    CK_TILE_HOST_DEVICE constexpr auto get_as(number<I>) const {TB_COMMON_AS();
            if constexpr(sizeof(value_type) <= 1 )
            return _get_as<Tx>(number<I>{});   // TODO: current compiler for 8bit data need use union to get data back, should fix in the future
            else
            return reinterpret_cast<const thread_buffer<Tx, vx>&>(data).get(number<I>{});}

    template <typename Tx> CK_TILE_HOST_DEVICE constexpr void set_as(index_t i, const Tx & x)
            { TB_COMMON_AS();    reinterpret_cast<thread_buffer<Tx, vx>&>(data).at(i) = x; }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr void set_as(number<I>, const Tx & x)
            { TB_COMMON_AS();    reinterpret_cast<thread_buffer<Tx, vx>&>(data).at(number<I>{}) = x; }

#undef TB_COMMON_AS
};
// clang-format on

template <typename, typename>
struct vector_traits;

// specialization for array
template <typename T, index_t N>
struct vector_traits<thread_buffer<T, N>, std::enable_if_t<!std::is_class_v<T>>>
{
    using scalar_type                    = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct vector_traits<thread_buffer<T, N>, std::enable_if_t<std::is_class_v<T>>>
{
    using scalar_type                    = typename T::type;
    static constexpr index_t vector_size = N;
};

#endif

template <char... Xs>
struct str_literal
{
    static constexpr const char data[] = {Xs..., '\0'};
    static constexpr const size_t size = sizeof...(Xs);

    template <char... Ys>
    CK_TILE_HOST_DEVICE constexpr auto operator+(str_literal<Ys...> /*rhs*/) const
    {
        return str_literal<Xs..., Ys...>{};
    }

    template <index_t N, char... Ys>
    CK_TILE_HOST_DEVICE static constexpr auto duplicate_n(const str_literal<Ys...> sep)
    {
        if constexpr(N == 0)
            return str_literal<>{};
        else if constexpr(N == 1)
            return str_literal<Xs...>{};
        else
            return duplicate_n<N - 1>(sep) + str_literal<Ys..., Xs...>{};
    }
};

#define make_str_literal(lit_)                                                                     \
    std::apply([](auto... indices) { return str_literal<(lit_)[decltype(indices)::value]...>{}; }, \
               makeTuple(std::make_index_sequence<constexpr_strlen(lit_)>()))

template <size_t... Idx>
constexpr std::tuple<std::integral_constant<size_t, Idx>...>
    makeTuple(std::index_sequence<Idx...>) noexcept
{
    return {};
}
constexpr size_t constexpr_strlen(const char* c)
{
    size_t t = 0;
    while(*c++)
        ++t;
    return t;
}
template <typename DataType_, typename StaticTileDistribution_>
struct static_distributed_tensor;

// Usage example: CK_PRINTF<float>{}(tensor);
template <typename ConvertTo = void,
          typename FMT       = str_literal<>,
          typename PREFIX    = str_literal<>,
          typename SUFFIX    = str_literal<>>
struct CK_PRINTF;
template <typename ConvertTo, char... FMTChars, char... PREFIXChars, char... SUFFIXChars>
struct CK_PRINTF<ConvertTo,
                 str_literal<FMTChars...>,
                 str_literal<PREFIXChars...>,
                 str_literal<SUFFIXChars...>>
{
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr auto default_format()
    {
        if constexpr(std::is_same_v<T, float>)
            return make_str_literal("%8.3f");
        else if constexpr(std::is_same_v<T, int>)
            return make_str_literal("%5d");
        else if constexpr(std::is_same_v<T, unsigned int>)
            return make_str_literal("%5u");
        else
            return make_str_literal("0x%08x");
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_prefix()
    {
        constexpr auto fmt_tid = make_str_literal("tid %03d: [%02d] ");
        if constexpr(sizeof...(PREFIXChars) == 0)
            return fmt_tid;
        else
            return fmt_tid + make_str_literal(" ") + str_literal<PREFIXChars...>{};
    }
    CK_TILE_HOST_DEVICE static constexpr auto get_suffix()
    {
        constexpr auto lf = make_str_literal("\n");
        if constexpr(sizeof...(SUFFIXChars) == 0)
            return lf;
        else
            return str_literal<SUFFIXChars...>{} + lf;
    }

    template <typename T, index_t N, typename Y, index_t... Is>
    CK_TILE_HOST_DEVICE void impl(const thread_buffer<T, N>& buf,
                                  std::integer_sequence<index_t, Is...>) const
    {
        using FMT1                = std::conditional_t<sizeof...(FMTChars) == 0,
                                        decltype(default_format<Y>()),
                                        str_literal<FMTChars...>>;
        constexpr auto fmt_v      = FMT1::template duplicate_n<N>(make_str_literal(" "));
        constexpr auto fmt_wrap_v = get_prefix() + fmt_v + get_suffix();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
        printf(fmt_wrap_v.data, get_thread_id(), N, type_convert<Y>(buf[Is])...);
#pragma clang diagnostic pop
    }

    template <typename T, index_t N>
    CK_TILE_HOST_DEVICE void operator()(const thread_buffer<T, N>& buf) const
    {
        using ConvertTo_ = std::conditional_t<std::is_same_v<ConvertTo, void>, T, ConvertTo>;
        impl<T, N, ConvertTo_>(buf, std::make_integer_sequence<index_t, N>{});
    }

    template <typename... TS>
    CK_TILE_HOST_DEVICE void operator()(const static_distributed_tensor<TS...>& tensor) const
    {
        return operator()(tensor.get_thread_buffer());
    }
};

template <typename ConvertTo = void,
          typename FMT       = str_literal<>,
          typename PREFIX    = str_literal<>,
          typename SUFFIX    = str_literal<>>
struct CK_PRINTF_T64 : public CK_PRINTF<ConvertTo, FMT, PREFIX, SUFFIX>
{
    using base_t = CK_PRINTF<ConvertTo, FMT, PREFIX, SUFFIX>;

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(const T& buf) const
    {
        if(get_thread_id() < 64)
            base_t::operator()(buf);
    }
};
} // namespace ck_tile
