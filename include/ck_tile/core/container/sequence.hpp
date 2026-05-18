// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/to_sequence.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

template <index_t...>
struct sequence;

template <typename Seq, index_t I>
struct sequence_split;

template <typename>
struct sequence_reverse;

template <typename>
struct sequence_map_inverse;

template <typename>
struct is_valid_sequence_map;

template <index_t I, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto sequence_pop_front(sequence<I, Is...>);

template <typename Seq>
CK_TILE_HOST_DEVICE constexpr auto sequence_pop_back(Seq);

// Implementation details for sequence element access and index generation.
namespace detail {

// O(1) type pack indexing via compiler builtin when available,
// with an O(N) recursive fallback for compilers that lack it (e.g., older MSVC).
// Does not depend on <tuple> so it is safe for hipRTC / GPU codegen.
#if defined(__has_builtin) && __has_builtin(__type_pack_element)
template <index_t I, typename... Ts>
using at_index_t = __type_pack_element<I, Ts...>;
#else
template <index_t I, typename T, typename... Ts>
struct type_pack_element_impl
{
    using type = typename type_pack_element_impl<I - 1, Ts...>::type;
};

template <typename T, typename... Ts>
struct type_pack_element_impl<0, T, Ts...>
{
    using type = T;
};

template <index_t I, typename... Ts>
using at_index_t = typename type_pack_element_impl<I, Ts...>::type;
#endif

// Bridge type for __make_integer_seq: converts integer pack to ck_tile::sequence
template <typename T, T... Ints>
struct integer_sequence_wrapper;

template <index_t... Ints>
struct integer_sequence_wrapper<index_t, Ints...>
{
    using seq_type = sequence<Ints...>;
};
} // namespace detail

template <index_t N>
using make_index_sequence =
    typename __make_integer_seq<detail::integer_sequence_wrapper, index_t, N>::seq_type;

template <index_t... Is>
struct sequence
{
    using type       = sequence;
    using value_type = index_t;

    CK_TILE_HOST_DEVICE static constexpr index_t size() { return sizeof...(Is); }
    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return true; };

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto get()
    {
        static_assert(I < size(), "wrong! I too large");
        return number<detail::at_index_t<I, constant<Is>...>{}>{};
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto get(number<I>)
    {
        static_assert(I < size(), "wrong! I too large");
        return number<get<I>()>{};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t at(index_t I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const index_t mData[size() + 1] = {Is..., 0};
        return mData[I];
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto at()
    {
        static_assert(I < size(), "wrong! I too large");
        return number<detail::at_index_t<I, constant<Is>...>{}>{};
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto at(number<I>)
    {
        static_assert(I < size(), "wrong! I too large");
        return number<get<I>()>{};
    }

    template <typename I>
    CK_TILE_HOST_DEVICE constexpr auto operator[](I i) const
    {
        return at(i);
    }

    template <index_t... IRs>
    CK_TILE_HOST_DEVICE static constexpr auto reorder_new_to_old(sequence<IRs...> /*new2old*/)
    {
        static_assert(sizeof...(Is) == sizeof...(IRs),
                      "wrong! reorder map should have the same size as sequence to be rerodered");

        static_assert(is_valid_sequence_map<sequence<IRs...>>::value, "wrong! invalid reorder map");

        return sequence<type::get(number<IRs>{})...>{};
    }

    // MapOld2New is sequence<...>
    template <typename MapOld2New>
    CK_TILE_HOST_DEVICE static constexpr auto reorder_old_to_new(MapOld2New)
    {
        static_assert(MapOld2New::size() == size(),
                      "wrong! reorder map should have the same size as sequence to be rerodered");

        static_assert(is_valid_sequence_map<remove_cvref_t<MapOld2New>>::value,
                      "wrong! invalid reorder map");

        return reorder_new_to_old(
            typename sequence_map_inverse<remove_cvref_t<MapOld2New>>::type{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto reverse()
    {
        return typename sequence_reverse<type>::type{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto front()
    {
        static_assert(size() > 0, "wrong!");
        return get(number<0>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto back()
    {
        static_assert(size() > 0, "wrong!");
        return get(number<size() - 1>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto pop_front() { return sequence_pop_front(type{}); }

    CK_TILE_HOST_DEVICE static constexpr auto pop_back() { return sequence_pop_back(type{}); }

    template <index_t... Xs>
    CK_TILE_HOST_DEVICE static constexpr auto push_front(sequence<Xs...>)
    {
        return sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    CK_TILE_HOST_DEVICE static constexpr auto push_front(number<Xs>...)
    {
        return sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    CK_TILE_HOST_DEVICE static constexpr auto push_back(sequence<Xs...>)
    {
        return sequence<Is..., Xs...>{};
    }

    template <index_t... Xs>
    CK_TILE_HOST_DEVICE static constexpr auto push_back(number<Xs>...)
    {
        return sequence<Is..., Xs...>{};
    }

    // pickup element at index <Ids...>
    template <index_t... Ids>
    CK_TILE_HOST_DEVICE static constexpr auto extract(number<Ids>...)
    {
        return sequence<type::get(number<Ids>{})...>{};
    }

    template <index_t... Ids>
    CK_TILE_HOST_DEVICE static constexpr auto extract(sequence<Ids...>)
    {
        return sequence<type::get(number<Ids>{})...>{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto sum() { return (Is + ... + 0); }
    CK_TILE_HOST_DEVICE static constexpr auto product() { return (Is * ... * 1); }

    // modify element at index "I" with value "X"
    template <index_t I, index_t X>
    CK_TILE_HOST_DEVICE static constexpr auto modify(number<I>, number<X>)
    {
        static_assert(I >= 0 && I < size(), "Index I is out of bounds");
        return modify_impl(make_index_sequence<size()>{}, number<I>{}, number<X>{});
    }

    private:
    template <index_t... Idxs, index_t ModifyIdx, index_t NewVal>
    CK_TILE_HOST_DEVICE static constexpr auto
    modify_impl(sequence<Idxs...>, number<ModifyIdx>, number<NewVal>)
    {
        return sequence<(Idxs == ModifyIdx ? NewVal : get<Idxs>())...>{};
    }

    public:
    template <typename F>
    CK_TILE_HOST_DEVICE static constexpr auto transform(F f)
    {
        return sequence<f(Is)...>{};
    }
};

template <index_t... Is>
CK_TILE_HOST_DEVICE static void print(const sequence<Is...>&)
{
    printf("sequence<");
    if constexpr(sizeof...(Is) > 0)
    {
        bool first = true;
        (([&first](index_t value) {
             printf("%s%d", first ? "" : ", ", value);
             first = false;
         }(Is)),
         ...);
    }
    printf(">");
}

template <typename T>
struct is_sequence : std::false_type
{
};
template <index_t... Is>
struct is_sequence<sequence<Is...>> : std::true_type
{
};
template <typename T>
inline constexpr bool is_sequence_v = is_sequence<T>::value;

// merge sequence
template <typename Seq, typename... Seqs>
struct sequence_merge
{
    using type = typename sequence_merge<Seq, typename sequence_merge<Seqs...>::type>::type;
};

template <index_t... Xs, index_t... Ys>
struct sequence_merge<sequence<Xs...>, sequence<Ys...>>
{
    using type = sequence<Xs..., Ys...>;
};

template <typename Seq>
struct sequence_merge<Seq>
{
    using type = Seq;
};

namespace detail {

// Bridge: converts __make_integer_seq index pack into a sequence via functor application.
template <typename T, T... Ids>
struct sequence_gen_helper
{
    template <typename F>
    using apply = sequence<F{}(number<Ids>{})...>;
};

} // namespace detail

/**
 * @brief Generate a compile-time sequence by applying a functor to indices 0..N-1.
 * @tparam NSize Number of elements in the generated sequence.
 * @tparam F Functor type; must be default-constructible with a constexpr call operator
 *         accepting number<I> (or index_t via implicit conversion) and returning index_t.
 *         Lambdas with captures cannot be used; use a template struct functor instead.
 */
template <index_t NSize, typename F>
struct sequence_gen
{
    using type =
        typename __make_integer_seq<detail::sequence_gen_helper, index_t, NSize>::template apply<F>;
};

template <typename F>
struct sequence_gen<0, F>
{
    using type = sequence<>;
};

// arithmetic sequence
template <index_t IBegin, index_t IEnd, index_t Increment>
struct arithmetic_sequence_gen
{
    struct F
    {
        CK_TILE_HOST_DEVICE constexpr index_t operator()(index_t i) const
        {
            return i * Increment + IBegin;
        }
    };

    using type0 = typename sequence_gen<(IEnd - IBegin) / Increment, F>::type;
    using type1 = sequence<>;

    static constexpr bool kHasContent =
        (Increment > 0 && IBegin < IEnd) || (Increment < 0 && IBegin > IEnd);

    using type = typename std::conditional<kHasContent, type0, type1>::type;
};

template <index_t IEnd>
struct arithmetic_sequence_gen<0, IEnd, 1>
{
    using type = make_index_sequence<IEnd>;
};

// uniform sequence - optimized using __make_integer_seq
namespace detail {

template <typename T, T... Ids>
struct uniform_sequence_helper
{
    // Comma operator: discard Ids, produce Value for each element
    template <index_t Value>
    using apply = sequence<((void)Ids, Value)...>;
};

} // namespace detail

template <index_t NSize, index_t I>
struct uniform_sequence_gen
{
    using type = typename __make_integer_seq<detail::uniform_sequence_helper, index_t, NSize>::
        template apply<I>;
};

template <index_t I>
struct uniform_sequence_gen<0, I>
{
    using type = sequence<>;
};

// inclusive scan (with init) sequence - optimized using constexpr for-loop with static_array
namespace detail {

template <typename Seq, typename Reduce, index_t Init, bool Reverse>
struct sequence_inclusive_scan_impl;

template <index_t... Is, typename Reduce, index_t Init, bool Reverse>
struct sequence_inclusive_scan_impl<sequence<Is...>, Reduce, Init, Reverse>
{
    template <index_t... Indices>
    static constexpr auto compute(sequence<Indices...>)
    {
        constexpr index_t size = sizeof...(Is);
        if constexpr(size == 0)
        {
            return sequence<>{};
        }
        else
        {
            // Compute all scan values in a single constexpr evaluation using
            // static_array, then unpack via index expansion. Avoids O(N) recursive
            // template instantiation.
            constexpr auto arr = []() {
                static_array<index_t, size> values = {Is...};
                static_array<index_t, size> result = {0};
                if constexpr(Reverse)
                {
                    // Reverse scan: right to left
                    result[size - 1] = Reduce{}(values[size - 1], Init);
                    for(index_t i = size - 1; i > 0; --i)
                    {
                        result[i - 1] = Reduce{}(values[i - 1], result[i]);
                    }
                }
                else
                {
                    // Forward scan: left to right
                    result[0] = Reduce{}(values[0], Init);
                    for(index_t i = 1; i < size; ++i)
                    {
                        result[i] = Reduce{}(values[i], result[i - 1]);
                    }
                }
                return result;
            }();
            return sequence<arr[Indices]...>{};
        }
    }

    using type = decltype(compute(make_index_sequence<sizeof...(Is)>{}));
};

// Exclusive scan: result[0] = Init, result[i] = Reduce(values[i-1], result[i-1]) for i > 0.
template <typename Seq, typename Reduce, index_t Init>
struct sequence_exclusive_scan_impl;

template <index_t... Is, typename Reduce, index_t Init>
struct sequence_exclusive_scan_impl<sequence<Is...>, Reduce, Init>
{
    template <index_t... Indices>
    static constexpr auto compute(sequence<Indices...>)
    {
        constexpr index_t size = sizeof...(Is);
        if constexpr(size == 0)
        {
            return sequence<>{};
        }
        else
        {
            constexpr auto arr = []() {
                static_array<index_t, size> values = {Is...};
                static_array<index_t, size> result = {0};
                result[0]                          = Init;
                for(index_t i = 1; i < size; ++i)
                {
                    result[i] = Reduce{}(values[i - 1], result[i - 1]);
                }
                return result;
            }();
            return sequence<arr[Indices]...>{};
        }
    }

    using type = decltype(compute(make_index_sequence<sizeof...(Is)>{}));
};

} // namespace detail

template <typename Seq, typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan
{
    using type = typename detail::sequence_inclusive_scan_impl<Seq, Reduce, Init, true>::type;
};

template <typename Seq, typename Reduce, index_t Init>
struct sequence_inclusive_scan
{
    using type = typename detail::sequence_inclusive_scan_impl<Seq, Reduce, Init, false>::type;
};

// split sequence
template <typename Seq, index_t I>
struct sequence_split
{
    static constexpr index_t NSize = Seq{}.size();

    using range0 = typename arithmetic_sequence_gen<0, I, 1>::type;
    using range1 = typename arithmetic_sequence_gen<I, NSize, 1>::type;

    using left_type  = decltype(Seq::extract(range0{}));
    using right_type = decltype(Seq::extract(range1{}));
};

namespace detail {
template <typename Id, index_t... Ns>
struct seq_reverse;

template <index_t... Ids, index_t... Ns>
struct seq_reverse<sequence<Ids...>, Ns...>
{
    template <index_t I>
    using element = detail::at_index_t<I, constant<Ns>...>;
    using type    = sequence<element<(sizeof...(Ns) - 1 - Ids)>::value...>;
};
} // namespace detail

template <index_t... Ns>
struct sequence_reverse<sequence<Ns...>>
    : detail::seq_reverse<make_index_sequence<sizeof...(Ns)>, Ns...>
{
};

// template <index_t... Ns>
// using sequence_reverse_t = typename sequence_reverse<Ns...>::type;

#if 1
template <typename Reduce, typename Seq, typename... Seqs>
struct sequence_reduce
{
    using type = typename sequence_reduce<Reduce,
                                          Seq,
                                          typename sequence_reduce<Reduce, Seqs...>::type>::type;
};

template <typename Reduce, index_t... Xs, index_t... Ys>
struct sequence_reduce<Reduce, sequence<Xs...>, sequence<Ys...>>
{
    using type = sequence<Reduce{}(Xs, Ys)...>;
};

template <typename Reduce, typename Seq>
struct sequence_reduce<Reduce, Seq>
{
    using type = Seq;
};
#endif

// Sorts a sequence using constexpr insertion sort. O(1) template instantiation
// depth, replacing the recursive merge sort that created O(N log N) intermediate types.
namespace detail {

template <typename Values, typename Compare, typename IndexSeq>
struct sequence_sort_helper;

template <index_t... Vs, typename Compare, index_t... Idx>
struct sequence_sort_helper<sequence<Vs...>, Compare, sequence<Idx...>>
{
    struct sort_result
    {
        static_array<index_t, sizeof...(Vs)> values;
        static_array<index_t, sizeof...(Vs)> ids;
    };

    static constexpr sort_result compute()
    {
        constexpr index_t n = sizeof...(Vs);
        sort_result r{{{Vs...}}, {{Idx...}}};
        // insertion sort — O(N^2) constexpr steps, O(1) template depth
        for(index_t i = 1; i < n; ++i)
        {
            for(index_t j = i; j > 0 && Compare{}(r.values[j], r.values[j - 1]); --j)
            {
                auto tv         = r.values[j];
                r.values[j]     = r.values[j - 1];
                r.values[j - 1] = tv;
                auto ti         = r.ids[j];
                r.ids[j]        = r.ids[j - 1];
                r.ids[j - 1]    = ti;
            }
        }
        return r;
    }

    static constexpr sort_result sorted = compute();
    using sorted_values                 = sequence<sorted.values[Idx]...>;
    using sorted_ids                    = sequence<sorted.ids[Idx]...>;
};

} // namespace detail

template <typename Values, typename Compare>
struct sequence_sort
{
    static constexpr index_t n = Values::size();
    using idx_seq              = make_index_sequence<n>;

    using helper = detail::sequence_sort_helper<remove_cvref_t<Values>, Compare, idx_seq>;

    using type                = typename helper::sorted_values;
    using sorted2unsorted_map = typename helper::sorted_ids;
};

template <typename Values, typename Less, typename Equal>
struct sequence_unique_sort
{
    template <typename RemainValues,
              typename RemainIds,
              typename UniquifiedValues,
              typename UniquifiedIds,
              typename Eq>
    struct sorted_sequence_uniquify_impl
    {
        static constexpr index_t current_value = RemainValues::front();
        static constexpr index_t current_id    = RemainIds::front();

        static constexpr bool is_unique_value = (current_value != UniquifiedValues::back());

        using new_remain_values = decltype(RemainValues::pop_front());
        using new_remain_ids    = decltype(RemainIds::pop_front());

        using new_uniquified_values =
            typename std::conditional<is_unique_value,
                                      decltype(UniquifiedValues::push_back(
                                          number<current_value>{})),
                                      UniquifiedValues>::type;

        using new_uniquified_ids =
            typename std::conditional<is_unique_value,
                                      decltype(UniquifiedIds::push_back(number<current_id>{})),
                                      UniquifiedIds>::type;

        using uniquify = sorted_sequence_uniquify_impl<new_remain_values,
                                                       new_remain_ids,
                                                       new_uniquified_values,
                                                       new_uniquified_ids,
                                                       Eq>;

        // this is output
        using uniquified_values = typename uniquify::uniquified_values;
        using uniquified_ids    = typename uniquify::uniquified_ids;
    };

    template <typename UniquifiedValues, typename UniquifiedIds, typename Eq>
    struct sorted_sequence_uniquify_impl<sequence<>,
                                         sequence<>,
                                         UniquifiedValues,
                                         UniquifiedIds,
                                         Eq>
    {
        using uniquified_values = UniquifiedValues;
        using uniquified_ids    = UniquifiedIds;
    };

    template <typename SortedValues, typename SortedIds, typename Eq>
    struct sorted_sequence_uniquify
    {
        using uniquify = sorted_sequence_uniquify_impl<decltype(SortedValues::pop_front()),
                                                       decltype(SortedIds::pop_front()),
                                                       sequence<SortedValues::front()>,
                                                       sequence<SortedIds::front()>,
                                                       Eq>;

        using uniquified_values = typename uniquify::uniquified_values;
        using uniquified_ids    = typename uniquify::uniquified_ids;
    };

    using sort          = sequence_sort<Values, Less>;
    using sorted_values = typename sort::type;
    using sorted_ids    = typename sort::sorted2unsorted_map;

    using uniquify = sorted_sequence_uniquify<sorted_values, sorted_ids, Equal>;

    // this is output
    using type                = typename uniquify::uniquified_values;
    using sorted2unsorted_map = typename uniquify::uniquified_ids;
};

// Validates that a sequence is a permutation of {0, 1, ..., N-1}.
// Uses a constexpr loop instead of instantiating sequence_sort.
namespace detail {

template <index_t... Is>
constexpr bool check_valid_sequence_map()
{
    constexpr index_t n = sizeof...(Is);
    if constexpr(n == 0)
    {
        return true;
    }
    else
    {
        constexpr index_t vals[] = {Is...};
        static_array<bool, n> seen{};
        for(index_t i = 0; i < n; ++i)
        {
            if(vals[i] < 0 || vals[i] >= n || seen[vals[i]])
                return false;
            seen[vals[i]] = true;
        }
        return true;
    }
}

} // namespace detail

template <typename SeqMap>
struct is_valid_sequence_map : std::false_type
{
};

template <index_t... Is>
struct is_valid_sequence_map<sequence<Is...>>
    : std::integral_constant<bool, detail::check_valid_sequence_map<Is...>()>
{
};

/**
 * @brief Compute the inverse permutation of a sequence map.
 * @tparam Is A valid permutation of {0, 1, ..., N-1}.
 * @pre Input must satisfy is_valid_sequence_map (enforced by static_assert).
 *
 * Optimized using constexpr for-loop: O(1) template instantiation depth instead of O(N).
 */
template <index_t... Is>
struct sequence_map_inverse<sequence<Is...>>
{
    static_assert(is_valid_sequence_map<sequence<Is...>>::value,
                  "sequence_map_inverse requires a valid permutation sequence map");

    private:
    static constexpr auto build_inverse()
    {
        static_assert(sizeof...(Is) > 0, "build_inverse requires non-empty sequence");
        static_array<index_t, sizeof...(Is)> result = {0};
        constexpr index_t input[]                   = {Is...};
        for(index_t pos = 0; pos < static_cast<index_t>(sizeof...(Is)); ++pos)
        {
            result[input[pos]] = pos;
        }
        return result;
    }

    static constexpr auto inverse = build_inverse();

    template <index_t... Positions>
    static constexpr auto compute(sequence<Positions...>)
    {
        return sequence<inverse[Positions]...>{};
    }

    public:
    using type = decltype(compute(make_index_sequence<sizeof...(Is)>{}));
};

template <>
struct sequence_map_inverse<sequence<>>
{
    using type = sequence<>;
};

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr bool operator==(sequence<Xs...>, sequence<Ys...>)
{
    return ((Xs == Ys) && ...);
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr bool operator!=(sequence<Xs...> x, sequence<Ys...> y)
{
    return !(x == y);
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator+(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs + Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator-(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs - Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator*(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs * Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator/(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs / Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator%(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs % Ys)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator+(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs + Y)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator-(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs - Y)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator*(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs * Y)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator/(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs / Y)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator%(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs % Y)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator+(number<Y>, sequence<Xs...>)
{
    return sequence<(Y + Xs)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator-(number<Y>, sequence<Xs...>)
{
    return sequence<(Y - Xs)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator*(number<Y>, sequence<Xs...>)
{
    return sequence<(Y * Xs)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator/(number<Y>, sequence<Xs...>)
{
    return sequence<(Y / Xs)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator%(number<Y>, sequence<Xs...>)
{
    return sequence<(Y % Xs)...>{};
}

template <index_t I, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto sequence_pop_front(sequence<I, Is...>)
{
    return sequence<Is...>{};
}

template <typename Seq>
CK_TILE_HOST_DEVICE constexpr auto sequence_pop_back(Seq)
{
    static_assert(Seq::size() > 0, "wrong! cannot pop an empty sequence!");
    return sequence_pop_front(Seq::reverse()).reverse();
}

template <typename... Seqs>
CK_TILE_HOST_DEVICE constexpr auto merge_sequences(Seqs...)
{
    return typename sequence_merge<Seqs...>::type{};
}

template <typename F, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto transform_sequences(F f, sequence<Xs...>)
{
    return sequence<f(Xs)...>{};
}

template <typename F, index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto transform_sequences(F f, sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sequence<Xs...>::size() == sequence<Ys...>::size(), "Dim not the same");

    return sequence<f(Xs, Ys)...>{};
}

template <typename F, index_t... Xs, index_t... Ys, index_t... Zs>
CK_TILE_HOST_DEVICE constexpr auto
transform_sequences(F f, sequence<Xs...>, sequence<Ys...>, sequence<Zs...>)
{
    static_assert(sequence<Xs...>::size() == sequence<Ys...>::size() &&
                      sequence<Xs...>::size() == sequence<Zs...>::size(),
                  "Dim not the same");

    return sequence<f(Xs, Ys, Zs)...>{};
}

template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr auto reverse_inclusive_scan_sequence(Seq, Reduce, number<Init>)
{
    return typename sequence_reverse_inclusive_scan<Seq, Reduce, Init>::type{};
}

template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr auto reverse_exclusive_scan_sequence(Seq, Reduce, number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq::pop_front(), Reduce{}, number<Init>{})
        .push_back(number<Init>{});
}

template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr auto inclusive_scan_sequence(Seq, Reduce, number<Init>)
{
    return typename sequence_inclusive_scan<Seq, Reduce, Init>::type{};
}

// e.g. Seq<2, 3, 4> --> Seq<0, 2, 5>, Init=0, Reduce=Add
template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr auto exclusive_scan_sequence(Seq, Reduce, number<Init>)
{
    return typename detail::sequence_exclusive_scan_impl<Seq, Reduce, Init>::type{};
}

// e.g. Seq<2, 3, 4> --> Seq<0, 2, 5, 9> (N+1 elements: prefix sums including both endpoints)
template <typename Seq>
CK_TILE_HOST_DEVICE constexpr auto prefix_sum_sequence(Seq)
{
    using extended = typename sequence_merge<Seq, sequence<0>>::type;
    return typename detail::sequence_exclusive_scan_impl<extended, plus<index_t>, 0>::type{};
}

template <typename Seq, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto pick_sequence_elements_by_ids(Seq, sequence<Is...> /* ids */)
{
    return sequence<Seq::get(number<Is>{})...>{};
}

#if 1
namespace detail {
template <typename WorkSeq, typename RemainSeq, typename RemainMask>
struct pick_sequence_elements_by_mask_impl
{
    using new_work_seq = typename std::conditional<RemainMask::front(),
                                                   decltype(WorkSeq::push_back(RemainSeq::front())),
                                                   WorkSeq>::type;

    using type =
        typename pick_sequence_elements_by_mask_impl<new_work_seq,
                                                     decltype(RemainSeq::pop_front()),
                                                     decltype(RemainMask::pop_front())>::type;
};

template <typename WorkSeq>
struct pick_sequence_elements_by_mask_impl<WorkSeq, sequence<>, sequence<>>
{
    using type = WorkSeq;
};

} // namespace detail

template <typename Seq, typename Mask>
CK_TILE_HOST_DEVICE constexpr auto pick_sequence_elements_by_mask(Seq, Mask)
{
    static_assert(Seq::size() == Mask::size(), "wrong!");

    return typename detail::pick_sequence_elements_by_mask_impl<sequence<>, Seq, Mask>::type{};
}

namespace detail {
template <typename WorkSeq, typename RemainValues, typename RemainIds>
struct modify_sequence_elements_by_ids_impl
{
    using new_work_seq = decltype(WorkSeq::modify(RemainIds::front(), RemainValues::front()));

    using type =
        typename modify_sequence_elements_by_ids_impl<new_work_seq,
                                                      decltype(RemainValues::pop_front()),
                                                      decltype(RemainIds::pop_front())>::type;
};

template <typename WorkSeq>
struct modify_sequence_elements_by_ids_impl<WorkSeq, sequence<>, sequence<>>
{
    using type = WorkSeq;
};
} // namespace detail

template <typename Seq, typename Values, typename Ids>
CK_TILE_HOST_DEVICE constexpr auto modify_sequence_elements_by_ids(Seq, Values, Ids)
{
    static_assert(Values::size() == Ids::size() && Seq::size() >= Values::size(), "wrong!");

    return typename detail::modify_sequence_elements_by_ids_impl<Seq, Values, Ids>::type{};
}
#endif

template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr index_t
reduce_on_sequence(Seq, Reduce f, number<Init> /*initial_value*/)
{
    index_t result = Init;

    for(index_t i = 0; i < Seq::size(); ++i)
    {
        result = f(result, Seq::at(i));
    }

    return result;
}

// TODO: a generic any_of for any container
template <typename Seq, typename F>
CK_TILE_HOST_DEVICE constexpr bool sequence_any_of(Seq, F f)
{
    bool flag = false;

    for(index_t i = 0; i < Seq::size(); ++i)
    {
        flag = flag || f(Seq::at(i));
    }

    return flag;
}

// TODO: a generic all_of for any container
template <typename Seq, typename F>
CK_TILE_HOST_DEVICE constexpr bool sequence_all_of(Seq, F f)
{
    bool flag = true;

    for(index_t i = 0; i < Seq::size(); ++i)
    {
        flag = flag && f(Seq::at(i));
    }

    return flag;
}

template <typename... Seqs>
using sequence_merge_t = typename sequence_merge<Seqs...>::type;

template <index_t NSize, index_t I>
using uniform_sequence_gen_t = typename uniform_sequence_gen<NSize, I>::type;

template <index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto make_sequence(number<Is>...)
{
    return sequence<Is...>{};
}

// F() returns index_t
// F use default constructor, so F cannot be lambda function
template <typename F, index_t N>
CK_TILE_HOST_DEVICE constexpr auto generate_sequence(F, number<N>)
{
    return typename sequence_gen<N, F>::type{};
}

// F() returns number<>
// F could be lambda function
template <typename F, index_t N>
CK_TILE_HOST_DEVICE constexpr auto generate_sequence_v2(F&& f, number<N>)
{
    return unpack([&f](auto&&... xs) { return make_sequence(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <class... T>
struct tuple;

template <index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto to_sequence(tuple<number<Is>...>)
{
    return sequence<Is...>{};
}

template <index_t... Is>
using number_tuple = tuple<number<Is>...>;
template <index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto to_number_tuple(sequence<Is...> = {})
{
    return number_tuple<Is...>{};
}

namespace detail {
template <index_t h_idx, typename SeqSortedSamples, typename SeqRange>
struct sorted_sequence_histogram;

template <index_t h_idx, index_t x, index_t... xs, index_t r, index_t... rs>
struct sorted_sequence_histogram<h_idx, sequence<x, xs...>, sequence<r, rs...>>
{
    template <typename Histogram>
    constexpr auto operator()(Histogram& h)
    {
        if constexpr(x < r)
        {
            h.template at<h_idx>() += 1;
            sorted_sequence_histogram<h_idx, sequence<xs...>, sequence<r, rs...>>{}(h);
        }
        else
        {
            h.template at<h_idx + 1>() = 1;
            sorted_sequence_histogram<h_idx + 1, sequence<xs...>, sequence<rs...>>{}(h);
        }
    }
};

template <index_t h_idx, index_t x, index_t r, index_t... rs>
struct sorted_sequence_histogram<h_idx, sequence<x>, sequence<r, rs...>>
{
    template <typename Histogram>
    constexpr auto operator()(Histogram& h)
    {
        if constexpr(x < r)
        {
            h.template at<h_idx>() += 1;
        }
    }
};
} // namespace detail

template <typename, index_t>
struct array; // declare for later use (array->seq utility)

// SeqSortedSamples: <0, 2, 3, 5, 7>, SeqRange: <0, 3, 6, 9> -> SeqHistogram : <2, 2, 1>
template <typename SeqSortedSamples, index_t r, index_t... rs>
CK_TILE_HOST_DEVICE constexpr auto histogram_sorted_sequence(SeqSortedSamples, sequence<r, rs...>)
{
    constexpr auto bins      = sizeof...(rs); // or categories
    constexpr auto histogram = [&]() {
        array<index_t, bins> h{0}; // make sure this can clear all element to zero
        detail::sorted_sequence_histogram<0, SeqSortedSamples, sequence<rs...>>{}(h);
        return h;
    }();

    return TO_SEQUENCE(histogram, bins);
}

template <typename F, index_t N>
CK_TILE_HOST_DEVICE constexpr auto generate_array(F&& f, number<N>)
{
    using T = remove_cvref_t<decltype(f(number<0>{}))>;

    return unpack([&f](auto&&... is) { return array<T, N>{f(is)...}; },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

namespace detail {
template <typename, typename, typename, index_t>
struct reverse_slice_sequence_impl;

template <index_t x,
          index_t... xs,
          index_t m,
          index_t... ms,
          index_t id,
          index_t... ids,
          index_t SliceSize>
struct reverse_slice_sequence_impl<sequence<x, xs...>,
                                   sequence<m, ms...>,
                                   sequence<id, ids...>,
                                   SliceSize>
{
    using old_scan =
        reverse_slice_sequence_impl<sequence<xs...>, sequence<ms...>, sequence<ids...>, SliceSize>;

    static constexpr auto slice_size = old_scan::remaining_slice_sizes::front().value;
    static constexpr auto slice_length =
        std::conditional_t<m, number<gcd(x, slice_size)>, number<x>>::value;

    using dim_lengths =
        typename sequence_merge<sequence<slice_length>, typename old_scan::dim_lengths>::type;
    using dim_slices =
        typename sequence_merge<sequence<x / slice_length>, typename old_scan::dim_slices>::type;
    using remaining_slice_sizes = typename sequence_merge<
        std::conditional_t<m, sequence<slice_size / slice_length>, sequence<slice_size>>,
        typename old_scan::remaining_slice_sizes>::type;

    // the first idx that sliced length not equal to original length
    static constexpr index_t _flag =
        slice_length != x && remaining_slice_sizes{}.front().value == 1;
    static constexpr index_t _split_flag = std::conditional_t<m, number<_flag>, number<0>>::value;
    static constexpr index_t _split_idx =
        std::conditional_t<_split_flag, number<id>, number<0>>::value;

    static constexpr index_t split_flag = _split_flag || old_scan::split_flag;
    static constexpr index_t split_idx  = std::
        conditional_t<old_scan::split_flag, number<old_scan::split_idx>, number<_split_idx>>::value;
};

template <index_t x, index_t m, index_t id, index_t SliceSize>
struct reverse_slice_sequence_impl<sequence<x>, sequence<m>, sequence<id>, SliceSize>
{
    static constexpr auto slice_size = SliceSize;
    static constexpr auto slice_length =
        std::conditional_t<m, number<gcd(x, slice_size)>, number<x>>::value;

    using dim_lengths = sequence<slice_length>;
    using dim_slices  = sequence<x / slice_length>;
    using remaining_slice_sizes =
        std::conditional_t<m, sequence<slice_size / slice_length>, sequence<slice_size>>;

    // the first idx that sliced length not equal to original length
    static constexpr index_t _flag =
        slice_length != x && remaining_slice_sizes{}.front().value == 1;
    static constexpr index_t split_flag = std::conditional_t<m, number<_flag>, number<0>>::value;
    static constexpr index_t split_idx =
        std::conditional_t<split_flag, number<id>, number<0>>::value;
};
} // namespace detail

// clang-format off
// input a sequence(with optional mask), and the SliceSize : size per slice
// output the sequence each slice, and number of slices
// the length count for slice size is from right to left(reverse slice)
// or we can say, find the greatest common divider(gcd) from right to left, for the slice length
//
// e.g. <2, 8, 4>, slice length = 16
//  step-1: we take the right most <*, *, 4>, remaining 16/4=4
//  step-2: we only need 4 out of 8, of the midden dim, hence <*, 4, 4>
//  step-3: since nonthing remain, so the first dim we only need 1, hence<1, 4, 4>
//  => we got <1, 4, 4> as length for each slice
//  => total number of slice = <2, 8, 4> / <1, 4, 4> = <2, 2, 1>
//
// e.g. <2, 1, 4, 2>, 8     -> lengths:<1, 1, 4, 2>    , nums: <2, 1, 1, 1>    : 2 slices  , slice_idx: 0
//      <4, 2, 4, 1, 2>, 4  -> lengths:<1, 1, 2, 1, 2> , nums: <4, 2, 2, 1, 1> : 16 slices , slice_idx: 2
//      <4, 2, 4, 1, 6>, 4  -> lengths:<1, 1, 2, 1, 2> , nums: <4, 2, 2, 1, 3> : 48 slices , slice_idx: 2
//      <4, 2, 5, 1, 2>, 10 -> lengths:<1, 1, 5, 1, 2> , nums: <4, 2, 1, 1, 1> : 8 slices  , slice_idx: 1
//
//      <4, 2, 8>, 64       -> lengths:<4, 2, 8>       , nums: <1, 1, 1>       : 1  slices , slice_idx: 0
//      <4, 2, 8>, 32       -> lengths:<2, 2, 8>       , nums: <2, 1, 1>       : 2  slices , slice_idx: 0
//      <4, 2, 8>, 16       -> lengths:<1, 2, 8>       , nums: <4, 1, 1>       : 4  slices , slice_idx: 0
//      <4, 2, 8>, 8        -> lengths:<1, 1, 8>       , nums: <4, 2, 1>       : 8  slices , slice_idx: 1
//      <4, 2, 8>, 4        -> lengths:<1, 1, 4>       , nums: <4, 2, 2>       : 16 slices , slice_idx: 2
//      <4, 2, 8>, 2        -> lengths:<1, 1, 2>       , nums: <4, 2, 4>       : 32 slices , slice_idx: 2
//      <4, 2, 8>, 1        -> lengths:<1, 1, 1>       , nums: <4, 2, 8>       : 64 slices , slice_idx: 2
//
//      <4, 2, 1, 4, 2> / 4 ->
// mask:<1, 1, 1, 0, 1>,    -> lengths:<1, 2, 1, 4, 2> , nums: <4, 1, 1, 1, 1> : 8 slices  , slice_idx: 0
//
// return tuple<slice_lengths, slice_nums, slice_index>, slice_index is at which index will start
// have split slices (right -> left)
//  or the first index (right -> left) that sliced length is different from the original length
// clang-format on
template <typename Seq,
          index_t SliceSize,
          typename Mask = typename uniform_sequence_gen<Seq::size(), 1>::type>
constexpr auto reverse_slice_sequence(Seq,
                                      number<SliceSize>,
                                      Mask = typename uniform_sequence_gen<Seq::size(), 1>::type{})
{
    static_assert(Seq::size() == Mask::size());
    static_assert(SliceSize != 0, "slice size zero is invalid");
    static_assert(
        container_reduce(pick_sequence_elements_by_mask(Seq{}, Mask{}), multiplies<>{}, 1) %
                SliceSize ==
            0,
        "slice size can't evenly divide input sizes");
    using sliced_type = detail::reverse_slice_sequence_impl<
        Seq,
        Mask,
        typename arithmetic_sequence_gen<0, Seq::size(), 1>::type,
        SliceSize>;
    static_assert(sliced_type::remaining_slice_sizes::front().value == 1,
                  "can not evenly divide this sequence, please check");
    return make_tuple(typename sliced_type::dim_lengths{},
                      typename sliced_type::dim_slices{},
                      number<sliced_type::split_idx>{});
}

template <typename Seq,
          index_t SliceSize,
          typename Mask = typename uniform_sequence_gen<Seq::size(), 1>::type>
constexpr auto
slice_sequence(Seq, number<SliceSize>, Mask = typename uniform_sequence_gen<Seq::size(), 1>::type{})
{
    constexpr auto r =
        reverse_slice_sequence(Seq{}.reverse(), number<SliceSize>{}, Mask{}.reverse());
    return make_tuple(r[number<0>{}].reverse(),
                      r[number<1>{}].reverse(),
                      number<Seq::size() - r[number<2>{}] - 1>{});
}

} // namespace ck_tile
