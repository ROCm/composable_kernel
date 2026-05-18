// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <ostream>
#endif

#include "ck/utility/integral_constant.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/functional.hpp"
#include "ck/utility/math.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

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

namespace detail {

/**
 * @brief Helper to generate integer sequences with custom Sequence class
 */
template <typename T, T... Ints>
struct __integer_sequence;

template <index_t... Ints>
struct __integer_sequence<index_t, Ints...>
{
    using seq_type = Sequence<Ints...>;
};

} // namespace detail

/**
 * @brief Generate a Sequence class with index_t integers from 0 to N-1
 * @tparam N The size of the sequence to generate
 */
template <index_t N>
using make_index_sequence =
    typename __make_integer_seq<detail::__integer_sequence, index_t, N>::seq_type;

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

    /**
     * @brief Modify the sequence at a specific index with a new value
     * @tparam I The index of the element to modify
     * @tparam X The new value to set at index I
     * @return A new Sequence with the value at index I replaced by X
     */
    template <index_t I, index_t X>
    __host__ __device__ static constexpr auto Modify(Number<I>, Number<X>)
    {
        // Generate and forward an index sequence that covers all elements
        static_assert(I >= 0 && I < mSize, "Index I is out of bounds");
        return modify_impl(make_index_sequence<mSize>{}, Number<I>{}, Number<X>{});
    }

    private:
    /**
     * @brief Helper function to modify the sequence at a specific index
     * @tparam Idxs Indices of the sequence elements (0, 1, ..., Size-1)
     * @tparam ModifyIdx The index of the value in the sequence to modify
     * @tparam NewVal The new value to set at ModifyIdx
     * @return A new Sequence with the value at ModifyIdx replaced by NewVal
     */
    template <index_t... Idxs, index_t ModifyIdx, index_t NewVal>
    __host__ __device__ static constexpr auto
    modify_impl(Sequence<Idxs...>, Number<ModifyIdx>, Number<NewVal>)
    {
        // For each index: if it equals ModifyIdx, use NewVal; otherwise use original value
        return Sequence<(Idxs == ModifyIdx ? NewVal : At(Idxs))...>{};
    }

    public:
    template <typename F>
    __host__ __device__ static constexpr auto Transform(F f)
    {
        return Sequence<f(Is)...>{};
    }

    __host__ __device__ static void Print()
    {
        printf("{");
        printf("size %d, ", index_t{Size()});
        static_for<0, Size(), 1>{}([&](auto i) { printf("%d ", At(i).value); });
        printf("}");
    }
};

// merge sequence - optimized to avoid recursive instantiation
//
// Note: Unlike sequence_gen and uniform_sequence_gen which use __make_integer_seq for O(1)
// instantiation depth, sequence_merge cannot achieve O(1) depth. Here's why:
//
// - sequence_gen and uniform_sequence_gen generate a SINGLE output sequence where each
//   element can be computed independently: output[i] = f(i)
//
// - sequence_merge takes MULTIPLE input sequences with different, unknown lengths.
//   To compute output[i], we need to know:
//   1. Which input sequence contains this index
//   2. The offset within that sequence
//   This requires computing cumulative sequence lengths, which requires recursion/iteration.
//
// Instead, we use a binary tree reduction approach that achieves O(log N) instantiation depth:
// - Base cases handle 1-4 sequences directly (O(1) for common cases)
// - Recursive case merges pairs then combines: merge(s1,s2) + merge(s3,s4,...)
// - This gives O(log N) depth, which is optimal for merging heterogeneous sequences
//
// Alternative considered: Fold expressions (... + sequences) would give O(N) depth due to
// linear dependency chain, so binary tree is superior.
//
namespace detail {

// Helper to concatenate multiple sequences in one step using fold expression
template <typename... Seqs>
struct sequence_merge_impl;

// Base case: single sequence
template <index_t... Is>
struct sequence_merge_impl<Sequence<Is...>>
{
    using type = Sequence<Is...>;
};

// Two sequences: direct concatenation
template <index_t... Xs, index_t... Ys>
struct sequence_merge_impl<Sequence<Xs...>, Sequence<Ys...>>
{
    using type = Sequence<Xs..., Ys...>;
};

// Three sequences: direct concatenation (avoids one level of recursion)
template <index_t... Xs, index_t... Ys, index_t... Zs>
struct sequence_merge_impl<Sequence<Xs...>, Sequence<Ys...>, Sequence<Zs...>>
{
    using type = Sequence<Xs..., Ys..., Zs...>;
};

// Four sequences: direct concatenation
template <index_t... As, index_t... Bs, index_t... Cs, index_t... Ds>
struct sequence_merge_impl<Sequence<As...>, Sequence<Bs...>, Sequence<Cs...>, Sequence<Ds...>>
{
    using type = Sequence<As..., Bs..., Cs..., Ds...>;
};

// General case: binary tree reduction (O(log N) depth instead of O(N))
template <typename S1, typename S2, typename S3, typename S4, typename... Rest>
struct sequence_merge_impl<S1, S2, S3, S4, Rest...>
{
    // Merge pairs first, then recurse
    using left  = typename sequence_merge_impl<S1, S2>::type;
    using right = typename sequence_merge_impl<S3, S4, Rest...>::type;
    using type  = typename sequence_merge_impl<left, right>::type;
};

} // namespace detail

template <typename... Seqs>
struct sequence_merge
{
    using type = typename detail::sequence_merge_impl<Seqs...>::type;
};

template <>
struct sequence_merge<>
{
    using type = Sequence<>;
};

// generate sequence - optimized using __make_integer_seq to avoid recursive instantiation
namespace detail {

// Helper that applies functor F to indices and produces a Sequence
// __make_integer_seq<sequence_gen_helper, index_t, N> produces sequence_gen_helper<index_t, 0, 1,
// ..., N-1>
template <typename T, T... Is>
struct sequence_gen_helper
{
    // Apply a functor F to all indices at once via pack expansion (O(1) depth)
    template <typename F>
    using apply = Sequence<F{}(Number<Is>{})...>;
};

} // namespace detail

template <index_t NSize, typename F>
struct sequence_gen
{
    using type =
        typename __make_integer_seq<detail::sequence_gen_helper, index_t, NSize>::template apply<F>;
};

template <typename F>
struct sequence_gen<0, F>
{
    using type = Sequence<>;
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

    using type0 = typename sequence_gen<(IEnd - IBegin) / Increment, F>::type;
    using type1 = Sequence<>;

    static constexpr bool kHasContent =
        (Increment > 0 && IBegin < IEnd) || (Increment < 0 && IBegin > IEnd);

    using type = typename conditional<kHasContent, type0, type1>::type;
};

template <index_t IEnd>
struct arithmetic_sequence_gen<0, IEnd, 1>
{
    using type = make_index_sequence<IEnd>;
};

// uniform sequence - optimized using __make_integer_seq
namespace detail {

template <typename T, T... Is>
struct uniform_sequence_helper
{
    // Apply a constant value to all indices via pack expansion
    template <index_t Value>
    using apply = Sequence<((void)Is, Value)...>;
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
    using type = Sequence<>;
};

namespace detail {

/**
 * @brief A simple fixed-size array to hold intermediate results during constexpr computation
 * @tparam N The size of the array
 */
template <index_t N>
struct index_array
{
    index_t data[N > 0 ? N : 1];

    __host__ __device__ constexpr index_t& operator[](index_t i) [[clang::lifetimebound]]
    {
        return data[i];
    }
    __host__ __device__ constexpr const index_t& operator[](index_t i) const
        [[clang::lifetimebound]]
    {
        return data[i];
    }
};

/**
 * @brief Compute the reverse inclusive scan of a sequence at compile time using a constexpr
 * function
 * @tparam Reduce The binary reduction functor
 * @tparam Init The initial value for the reduction
 * @tparam Vs The input sequence values
 * @return An index_array containing the reverse inclusive scan results
 */
template <typename Reduce, index_t Init, index_t... Vs>
__host__ __device__ constexpr auto compute_reverse_inclusive_scan()
{
    constexpr index_t N = sizeof...(Vs);
    index_array<N> result{};
    constexpr index_t input[N > 0 ? N : 1] = {Vs...};

    if constexpr(N > 0)
    {
        result.data[N - 1] = Reduce{}(input[N - 1], Init);
        for(index_t i = N - 2; i >= 0; --i)
        {
            result.data[i] = Reduce{}(input[i], result.data[i + 1]);
        }
    }
    return result;
}

// Build result sequence with O(1) instantiation depth
template <typename Reduce, index_t Init, typename Seq, typename IndexSeq>
struct build_reverse_inclusive_scan;

template <typename Reduce, index_t Init, index_t... Vs, index_t... Is>
struct build_reverse_inclusive_scan<Reduce, Init, Sequence<Vs...>, Sequence<Is...>>
{
    static constexpr auto result = compute_reverse_inclusive_scan<Reduce, Init, Vs...>();

    using type = Sequence<result.data[Is]...>;
};

} // namespace detail

/**
 * @brief Reverse inclusive scan of a sequence - main interface
 * @tparam Seq The input sequence to scan
 * @tparam Reduce The binary reduction functor
 * @tparam Init The initial value for the reduction
 */
template <typename Seq, typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan
{
    using type = typename detail::
        build_reverse_inclusive_scan<Reduce, Init, Seq, make_index_sequence<Seq::Size()>>::type;
};

/**
 * @brief Specialization for empty sequence - returns empty sequence without computation
 * @tparam Reduce The binary reduction functor
 * @tparam Init The initial value for the reduction
 */
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

    using left_type  = decltype(Seq::Extract(range0{}));
    using right_type = decltype(Seq::Extract(range1{}));
};

// reverse sequence - optimized using direct pack expansion O(1) depth
namespace detail {

template <typename Seq, typename IndexSeq>
struct sequence_reverse_impl;

template <index_t... Is, index_t... Idxs>
struct sequence_reverse_impl<Sequence<Is...>, Sequence<Idxs...>>
{
    static constexpr index_t N = sizeof...(Is);
    // Access elements in reverse order: index (N-1-i) for position i
    using type = Sequence<Sequence<Is...>::At(Number<N - 1 - Idxs>{})...>;
};

} // namespace detail

template <typename Seq>
struct sequence_reverse
{
    using type =
        typename detail::sequence_reverse_impl<Seq, make_index_sequence<Seq::Size()>>::type;
};

// Empty sequence specialization
template <>
struct sequence_reverse<Sequence<>>
{
    using type = Sequence<>;
};

#if 1
template <typename Reduce, typename Seq, typename... Seqs>
struct sequence_reduce
{
    using type = typename sequence_reduce<Reduce,
                                          Seq,
                                          typename sequence_reduce<Reduce, Seqs...>::type>::type;
};

template <typename Reduce, index_t... Xs, index_t... Ys>
struct sequence_reduce<Reduce, Sequence<Xs...>, Sequence<Ys...>>
{
    using type = Sequence<Reduce{}(Xs, Ys)...>;
};

template <typename Reduce, typename Seq>
struct sequence_reduce<Reduce, Seq>
{
    using type = Seq;
};
#endif

// Implement sequence_sort and sequence_unique_sort using constexpr functions (C++17)
namespace sort_impl {

// Temporary arrays to hold values during operations with capacity N and mutable size.
template <index_t N>
struct IndexedValueArray
{
    index_t values[N > 0 ? N : 1];
    index_t ids[N > 0 ? N : 1];
    index_t size = 0;
};

template <index_t... Is>
constexpr auto make_indexed_value_array(Sequence<Is...>)
{
    constexpr index_t N         = sizeof...(Is);
    IndexedValueArray<N> result = {{Is...}, {}, N};
    for(index_t i = 0; i < N; ++i)
    {
        result.ids[i] = i;
    }
    return result;
}

enum class SortField
{
    Values,
    Ids
};

// Perform an insertion sort on an IndexedValueArray.
template <index_t N, typename Compare>
constexpr auto insertion_sort(IndexedValueArray<N> arr, Compare comp)
{
    for(index_t i = 1; i < arr.size; ++i)
    {
        index_t key_val = arr.values[i];
        index_t key_id  = arr.ids[i];
        index_t j       = i - 1;
        while(j >= 0 && comp(key_val, arr.values[j]))
        {
            arr.values[j + 1] = arr.values[j];
            arr.ids[j + 1]    = arr.ids[j];
            --j;
        }
        arr.values[j + 1] = key_val;
        arr.ids[j + 1]    = key_id;
    }
    return arr;
}

// Remove duplicates from a sorted IndexedValueArray.
template <index_t N, typename Equal>
constexpr auto unique(const IndexedValueArray<N>& sorted, Equal eq)
{
    IndexedValueArray<N> result{};
    if constexpr(N == 0)
    {
        return result;
    }
    result.size      = 1;
    result.values[0] = sorted.values[0];
    result.ids[0]    = sorted.ids[0];
    for(index_t i = 1; i < sorted.size; ++i)
    {
        if(!eq(sorted.values[i], sorted.values[i - 1]))
        {
            result.values[result.size] = sorted.values[i];
            result.ids[result.size]    = sorted.ids[i];
            ++result.size;
        }
    }
    return result;
}

// Compute sorted (and optionally unique) IndexedValueArray from input Sequence.
template <bool Unique, typename Compare, typename Equal, index_t... Is>
constexpr auto compute_sorted(Sequence<Is...> seq, Compare comp, Equal eq)
{
    auto sorted = insertion_sort(make_indexed_value_array(seq), comp);
    return Unique ? unique(sorted, eq) : sorted;
}

// Cache the sorted results to avoid recomputation.
template <bool Unique, typename Seq, typename Compare, typename Equal>
struct SortedCache
{
    static constexpr auto data = compute_sorted<Unique>(Seq{}, Compare{}, Equal{});
};

// Build sorted value and ID sequences from cached sorted data
template <SortField Field, bool Unique, typename Seq, typename Compare, typename Equal, index_t I>
constexpr index_t get_sorted_field()
{
    constexpr auto& data = SortedCache<Unique, Seq, Compare, Equal>::data;
    return (Field == SortField::Values) ? data.values[I] : data.ids[I];
}

template <bool Unique, typename Seq, typename Compare, typename Equal, typename IndexSeq>
struct SortedSequences;

template <bool Unique, typename Seq, typename Compare, typename Equal, index_t... Is>
struct SortedSequences<Unique, Seq, Compare, Equal, Sequence<Is...>>
{
    using values_type =
        Sequence<get_sorted_field<SortField::Values, Unique, Seq, Compare, Equal, Is>()...>;
    using ids_type =
        Sequence<get_sorted_field<SortField::Ids, Unique, Seq, Compare, Equal, Is>()...>;
};

template <bool Unique, typename Seq, typename Compare, typename Equal>
using sorted_sequences_t = SortedSequences<
    Unique,
    Seq,
    Compare,
    Equal,
    typename arithmetic_sequence_gen<0, SortedCache<Unique, Seq, Compare, Equal>::data.size, 1>::
        type>;

using Equal = ck::math::equal<index_t>;

} // namespace sort_impl

template <typename Values, typename Compare>
struct sequence_sort
{
    using sorted_seqs = sort_impl::sorted_sequences_t<false, Values, Compare, sort_impl::Equal>;
    using type        = typename sorted_seqs::values_type;
    using sorted2unsorted_map = typename sorted_seqs::ids_type;
};

template <typename Values, typename Less, typename Equal>
struct sequence_unique_sort
{
    using sorted_seqs         = sort_impl::sorted_sequences_t<true, Values, Less, Equal>;
    using type                = typename sorted_seqs::values_type;
    using sorted2unsorted_map = typename sorted_seqs::ids_type;
};

template <typename SeqMap>
struct is_valid_sequence_map : is_same<typename arithmetic_sequence_gen<0, SeqMap::Size(), 1>::type,
                                       typename sequence_sort<SeqMap, math::less<index_t>>::type>
{
};

/**
 * @brief  Invert a permutation sequence: given X2Y = {a, b, c, ...}, compute Y2X where Y2X[X2Y[i]]
 * = i Example: Sequence<2,0,1> (meaning pos0->2, pos1->0, pos2->1) inverts to Sequence<1,2,0>
 *
 * Why this implementation is faster to compile than recursive templates:
 *
 * The old recursive approach created a new template type for each element:
 *   sequence_map_inverse<Seq<2,0,1>> -> sequence_map_inverse<Seq<0,1>> ->
 *   sequence_map_inverse<Seq<1>>
 * Each "->" is a new type the compiler must create, track, and manage. For N elements, that's
 * N template types, each with overhead (name mangling, debug info, symbol table entries).
 *
 * This implementation uses a constexpr for loop to build the inverse in O(N) operations:
 * For input Sequence<2,0,1>, the loop sets result[input[pos]] = pos for each position:
 *   pos=0: result[2]=0, pos=1: result[0]=1, pos=2: result[1]=2
 * This builds the inverse permutation in a single pass with O(1) template instantiation depth.
 *
 * @tparam Is The input permutation sequence
 */
template <index_t... Is>
struct sequence_map_inverse<Sequence<Is...>>
{
    static_assert(is_valid_sequence_map<Sequence<Is...>>::value,
                  "sequence_map_inverse requires SeqMap to be a valid permutation sequence map");

    private:
    static constexpr auto build_inverse()
    {
        detail::index_array<sizeof...(Is)> result{};
        constexpr index_t input[] = {Is...};
        for(index_t pos = 0; pos < static_cast<index_t>(sizeof...(Is)); ++pos)
        {
            result[input[pos]] = pos;
        }
        return result;
    }

    static constexpr auto inverse = build_inverse();

    template <index_t... Positions>
    static constexpr auto compute(Sequence<Positions...>)
    {
        return Sequence<inverse[Positions]...>{};
    }

    public:
    using type = decltype(compute(make_index_sequence<sizeof...(Is)>{}));
};

template <>
struct sequence_map_inverse<Sequence<>>
{
    using type = Sequence<>;
};

template <index_t... Xs, index_t... Ys>
__host__ __device__ constexpr bool operator==(Sequence<Xs...>, Sequence<Ys...>)
{
    return ((Xs == Ys) && ...);
}

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

template <typename... Seqs>
__host__ __device__ constexpr auto merge_sequences(Seqs...)
{
    return typename sequence_merge<Seqs...>::type{};
}

template <typename F, index_t... Xs>
__host__ __device__ constexpr auto transform_sequences(F f, Sequence<Xs...>)
{
    return Sequence<f(Xs)...>{};
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
__host__ __device__ constexpr auto reverse_exclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq::PopFront(), Reduce{}, Number<Init>{})
        .PushBack(Number<Init>{});
}

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr auto inclusive_scan_sequence(Seq, Reduce, Number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq{}.Reverse(), Reduce{}, Number<Init>{}).Reverse();
}

template <typename Seq, index_t... Is>
__host__ __device__ constexpr auto pick_sequence_elements_by_ids(Seq, Sequence<Is...> /* ids */)
{
    return Sequence<Seq::At(Number<Is>{})...>{};
}

namespace detail {

template <typename WorkSeq, typename RemainSeq, typename RemainMask>
struct pick_sequence_elements_by_mask_impl
{
    using new_work_seq = typename conditional<RemainMask::Front(),
                                              decltype(WorkSeq::PushBack(RemainSeq::Front())),
                                              WorkSeq>::type;

    using type =
        typename pick_sequence_elements_by_mask_impl<new_work_seq,
                                                     decltype(RemainSeq::PopFront()),
                                                     decltype(RemainMask::PopFront())>::type;
};

template <typename WorkSeq>
struct pick_sequence_elements_by_mask_impl<WorkSeq, Sequence<>, Sequence<>>
{
    using type = WorkSeq;
};

} // namespace detail

template <typename Seq, typename Mask>
__host__ __device__ constexpr auto pick_sequence_elements_by_mask(Seq, Mask)
{
    static_assert(Seq::Size() == Mask::Size(), "wrong!");

    return typename detail::pick_sequence_elements_by_mask_impl<Sequence<>, Seq, Mask>::type{};
}

namespace detail {
template <typename WorkSeq, typename RemainValues, typename RemainIds>
struct modify_sequence_elements_by_ids_impl
{
    using new_work_seq = decltype(WorkSeq::Modify(RemainIds::Front(), RemainValues::Front()));

    using type =
        typename modify_sequence_elements_by_ids_impl<new_work_seq,
                                                      decltype(RemainValues::PopFront()),
                                                      decltype(RemainIds::PopFront())>::type;
};

template <typename WorkSeq>
struct modify_sequence_elements_by_ids_impl<WorkSeq, Sequence<>, Sequence<>>
{
    using type = WorkSeq;
};
} // namespace detail

template <typename Seq, typename Values, typename Ids>
__host__ __device__ constexpr auto modify_sequence_elements_by_ids(Seq, Values, Ids)
{
    static_assert(Values::Size() == Ids::Size() && Seq::Size() >= Values::Size(), "wrong!");

    return typename detail::modify_sequence_elements_by_ids_impl<Seq, Values, Ids>::type{};
}

template <typename Seq, typename Reduce, index_t Init>
__host__ __device__ constexpr index_t
reduce_on_sequence(Seq, Reduce f, Number<Init> /*initial_value*/)
{
    index_t result = Init;

    for(index_t i = 0; i < Seq::Size(); ++i)
    {
        result = f(result, Seq::At(i));
    }

    return result;
}

// TODO: a generic any_of for any container
template <typename Seq, typename F>
__host__ __device__ constexpr bool sequence_any_of(Seq, F f)
{
    bool flag = false;

    for(index_t i = 0; i < Seq::Size(); ++i)
    {
        flag = flag || f(Seq::At(i));
    }

    return flag;
}

// TODO: a generic all_of for any container
template <typename Seq, typename F>
__host__ __device__ constexpr bool sequence_all_of(Seq, F f)
{
    bool flag = true;

    for(index_t i = 0; i < Seq::Size(); ++i)
    {
        flag = flag && f(Seq::At(i));
    }

    return flag;
}

template <typename Sx, typename Sy>
using sequence_merge_t = typename sequence_merge<Sx, Sy>::type;

template <index_t NSize, index_t I>
using uniform_sequence_gen_t = typename uniform_sequence_gen<NSize, I>::type;

} // namespace ck

#pragma clang diagnostic pop

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
template <ck::index_t... Is>
std::ostream& operator<<(std::ostream& os, const ck::Sequence<Is...>)
{
    using S = ck::Sequence<Is...>;
    os << "{";
    ck::static_for<0, S::Size() - ck::Number<1>{}, 1>{}(
        [&](auto i) { os << S::At(i).value << ", "; });
    os << S::At(S::Size() - ck::Number<1>{}).value << "}";
    return os;
}
#endif
