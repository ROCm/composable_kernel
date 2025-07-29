// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <stdio.h>
#include <tuple>
#include <utility>

#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {
template <auto... val>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT()
{
}
template <typename... type>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT()
{
}

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

template <typename T_, index_t N_>
struct thread_buffer;

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
struct CK_PRINTF_WARP0 : public CK_PRINTF<ConvertTo, FMT, PREFIX, SUFFIX>
{
    using base_t = CK_PRINTF<ConvertTo, FMT, PREFIX, SUFFIX>;

    template <typename T>
    CK_TILE_HOST_DEVICE void operator()(const T& buf) const
    {
        if(get_thread_id() < get_warp_size())
            base_t::operator()(buf);
    }
};
} // namespace ck_tile
