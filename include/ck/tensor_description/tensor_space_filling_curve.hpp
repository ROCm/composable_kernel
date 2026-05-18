// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/statically_indexed_array_multi_index.hpp"
#include "ck/utility/tuple_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck {

namespace detail {

// Lookup table to store precomputed indices for all 1D access values
template <index_t NumAccesses, index_t nDim>
struct IndexLookupTable
{
    MultiIndex<nDim> data[NumAccesses > 0 ? NumAccesses : 1];

    __host__ __device__ constexpr const MultiIndex<nDim>& operator[](index_t i) const
        [[clang::lifetimebound]]
    {
        return data[i];
    }
};

// Compute a single index given 1D access index - used internally during table construction
// Uses static_for to work with MultiIndex which requires Number<I> for indexing
template <index_t nDim, bool SnakeCurved, typename Strides, typename OrderedAccessLengths>
__host__ __device__ constexpr auto
compute_single_index(index_t idx_1d, Strides strides, OrderedAccessLengths ordered_lengths)
{
    // Step 1: Convert 1D index to N-D ordered coordinates using strides
    MultiIndex<nDim> ordered_access_idx;
    index_t remaining = idx_1d;
    static_for<0, nDim, 1>{}([&](auto i) {
        ordered_access_idx(i) = remaining / strides[i];
        remaining             = remaining % strides[i];
    });

    // Step 2: Compute forward_sweep - whether each dimension is in forward direction
    StaticallyIndexedArray<bool, nDim> forward_sweep;
    forward_sweep(Number<0>{}) = true;
    index_t cumulative         = ordered_access_idx[Number<0>{}];
    static_for<1, nDim, 1>{}([&](auto i) {
        forward_sweep(i) = cumulative % 2 == 0;
        cumulative       = cumulative * ordered_lengths[i] + ordered_access_idx[i];
    });

    // Step 3: Apply snake curve transformation
    MultiIndex<nDim> ordered_idx;
    static_for<0, nDim, 1>{}([&](auto i) {
        if(!SnakeCurved || forward_sweep[i])
        {
            ordered_idx(i) = ordered_access_idx[i];
        }
        else
        {
            ordered_idx(i) = ordered_lengths[i] - 1 - ordered_access_idx[i];
        }
    });

    return ordered_idx;
}

// Precompute all indices into a lookup table using a single constexpr loop
template <index_t NumAccesses,
          index_t nDim,
          bool SnakeCurved,
          typename Strides,
          typename OrderedAccessLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess>
__host__ __device__ constexpr auto compute_all_indices(Strides strides,
                                                       OrderedAccessLengths ordered_lengths,
                                                       DimAccessOrder dim_order,
                                                       ScalarsPerAccess scalars)
{
    IndexLookupTable<NumAccesses, nDim> table{};

    for(index_t idx_1d = 0; idx_1d < NumAccesses; ++idx_1d)
    {
        auto ordered_idx =
            compute_single_index<nDim, SnakeCurved>(idx_1d, strides, ordered_lengths);

        // Reorder and scale
        auto reordered     = container_reorder_given_old2new(ordered_idx, dim_order);
        table.data[idx_1d] = reordered * scalars;
    }

    return table;
}

} // namespace detail

template <typename TensorLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess,
          bool SnakeCurved = true> // # of scalars per access in each dimension
struct SpaceFillingCurve
{
    static constexpr index_t nDim = TensorLengths::Size();

    using Index = MultiIndex<nDim>;

    static constexpr index_t ScalarPerVector =
        reduce_on_sequence(ScalarsPerAccess{}, math::multiplies{}, Number<1>{});

    static constexpr auto access_lengths   = TensorLengths{} / ScalarsPerAccess{};
    static constexpr auto dim_access_order = DimAccessOrder{};
    static constexpr auto ordered_access_lengths =
        container_reorder_given_new2old(access_lengths, dim_access_order);

    // Precompute access strides at class level
    static constexpr auto access_strides =
        container_reverse_exclusive_scan(ordered_access_lengths, math::multiplies{}, Number<1>{});

    // Number of access indices
    static constexpr index_t NumAccesses =
        reduce_on_sequence(TensorLengths{}, math::multiplies{}, Number<1>{}) / ScalarPerVector;

    // Precompute ALL indices into a lookup table - computed once at class instantiation
    static constexpr auto index_table = detail::compute_all_indices<NumAccesses, nDim, SnakeCurved>(
        access_strides, ordered_access_lengths, dim_access_order, ScalarsPerAccess{});

    static constexpr auto to_index_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(ordered_access_lengths)),
        make_tuple(typename arithmetic_sequence_gen<0, nDim, 1>::type{}),
        make_tuple(Sequence<0>{}));

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr index_t GetNumOfAccess()
    {
        static_assert(TensorLengths::Size() == ScalarsPerAccess::Size());
        static_assert(TensorLengths{} % ScalarsPerAccess{} ==
                      typename uniform_sequence_gen<TensorLengths::Size(), 0>::type{});

        return reduce_on_sequence(TensorLengths{}, math::multiplies{}, Number<1>{}) /
               ScalarPerVector;
    }

    template <index_t AccessIdx1dBegin, index_t AccessIdx1dEnd>
    static __device__ __host__ constexpr auto GetStepBetween(Number<AccessIdx1dBegin>,
                                                             Number<AccessIdx1dEnd>)
    {
        static_assert(AccessIdx1dBegin >= 0, "1D index should be non-negative");
        static_assert(AccessIdx1dBegin < GetNumOfAccess(), "1D index should be larger than 0");
        static_assert(AccessIdx1dEnd >= 0, "1D index should be non-negative");
        static_assert(AccessIdx1dEnd < GetNumOfAccess(), "1D index should be larger than 0");

        constexpr auto idx_begin = GetIndex(Number<AccessIdx1dBegin>{});
        constexpr auto idx_end   = GetIndex(Number<AccessIdx1dEnd>{});
        return idx_end - idx_begin;
    }

    template <index_t AccessIdx1d>
    static __device__ __host__ constexpr auto GetForwardStep(Number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d < GetNumOfAccess(), "1D index should be larger than 0");
        return GetStepBetween(Number<AccessIdx1d>{}, Number<AccessIdx1d + 1>{});
    }

    template <index_t AccessIdx1d>
    static __device__ __host__ constexpr auto GetBackwardStep(Number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d > 0, "1D index should be larger than 0");

        return GetStepBetween(Number<AccessIdx1d>{}, Number<AccessIdx1d - 1>{});
    }

    template <index_t AccessIdx1d>
    static __device__ __host__ constexpr Index GetIndex(Number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d >= 0 && AccessIdx1d < NumAccesses, "Index out of bounds");
        // Simple lookup from precomputed table - O(1) with no template instantiation overhead
        return index_table[AccessIdx1d];
    }

    // FIXME: rename this function
    template <index_t AccessIdx1d>
    static __device__ __host__ constexpr auto GetIndexTupleOfNumber(Number<AccessIdx1d>)
    {
        constexpr auto idx = GetIndex(Number<AccessIdx1d>{});

        return generate_tuple([&](auto i) { return Number<idx[i]>{}; }, Number<nDim>{});
    }
};

} // namespace ck

#pragma clang diagnostic pop
