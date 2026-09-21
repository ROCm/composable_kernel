// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/functional.hpp"
#include "ck/utility/functional2.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/multi_index.hpp"
#include "ck/utility/math.hpp"

namespace ck {

namespace detail {

/**
 * @brief Common base class for static_ford and ford.
 *
 * Provides shared compile-time constants and type aliases used by both
 * static_ford (compile-time iteration) and ford (runtime iteration).
 *
 * @tparam Lengths Sequence<L0, L1, ...> specifying the size of each dimension
 * @tparam Orders Sequence<O0, O1, ...> specifying the iteration order of dimensions.
 *                Orders[i] indicates which dimension is iterated at loop level i.
 */
template <class Lengths, class Orders>
struct ford_base
{
    /// Number of dimensions
    static constexpr index_t NDim = Lengths::Size();

    /// Total number of iterations (product of all lengths)
    static constexpr index_t TotalSize =
        reduce_on_sequence(Lengths{}, math::multiplies{}, Number<1>{});

    /// Lengths reordered according to iteration order
    static constexpr auto OrderedLengths = Lengths::ReorderGivenNew2Old(Orders{});

    /// Type of OrderedLengths with cv-qualifiers removed
    using OrderedLengthsType = remove_cvref_t<decltype(OrderedLengths)>;

    /// Mapping from loop level ("new" index) to original dimension ("old" index)
    using New2Old = Orders;

    __host__ __device__ constexpr ford_base()
    {
        static_assert(Lengths::GetSize() > 0, "wrong! Lengths is empty");
        static_assert(Lengths::GetSize() == Orders::GetSize(), "wrong! inconsistent size");
    }
};

/**
 * @brief Helper for decomposing a linear index into multi-dimensional indices.
 *
 * Computes strides at compile time and provides both compile-time and runtime
 * index decomposition. Used by static_ford and ford to convert a flat iteration
 * index into N-dimensional coordinates.
 *
 * For OrderedLengths = Sequence<L0, L1, L2>:
 *   - strides = {L1*L2, L2, 1}
 *   - ordered_idx[i] = (linear_idx / strides[i]) % lengths[i]
 *
 * @tparam OrderedLengths Sequence<...> of dimension sizes in iteration order
 * @tparam IndexSeq Sequence<0, 1, ..., NDim-1> for pack expansion
 */
template <class OrderedLengths, class IndexSeq>
struct index_decomposer;

template <index_t... Ls, index_t... Is>
struct index_decomposer<Sequence<Ls...>, Sequence<Is...>>
{
    /// Number of dimensions
    static constexpr index_t NDim = sizeof...(Ls);

    /// Dimension lengths in iteration order
    static constexpr index_array<NDim> lengths = {{Ls...}};

    /**
     * @brief Compute all strides in a single O(N) pass.
     *
     * For dimensions with lengths [L0, L1, L2, ...]:
     *   strides[N-1] = 1
     *   strides[i] = strides[i+1] * lengths[i+1]
     *
     * @return index_array containing computed strides
     */
    static constexpr index_array<NDim> compute_all_strides()
    {
        index_array<NDim> result{};
        if constexpr(NDim > 0)
        {
            result[NDim - 1] = 1;
            for(index_t i = NDim - 2; i >= 0; --i)
            {
                result[i] = result[i + 1] * lengths[i + 1];
            }
        }
        return result;
    }

    /// Pre-computed strides for each dimension
    static constexpr index_array<NDim> strides = compute_all_strides();

    /**
     * @brief Compile-time decomposition of a linear index.
     *
     * Returns a Sequence containing the multi-dimensional indices
     * in iteration order.
     *
     * @tparam LinearIdx The linear index to decompose (compile-time constant)
     */
    template <index_t LinearIdx>
    using decompose = Sequence<((LinearIdx / strides[Is]) % lengths[Is])...>;

    /**
     * @brief Runtime decomposition of a linear index with reordering.
     *
     * Decomposes linear_idx into ordered indices, then reorders them
     * to the original dimension order and stores in result.
     *
     * @tparam New2Old Sequence mapping iteration position to original dimension
     * @tparam MultiIndex Type of the output multi-index container
     * @param linear_idx The linear index to decompose
     * @param[out] result Multi-index container to store the result
     */
    template <class New2Old, class MultiIndex>
    __host__ __device__ static void decompose_runtime(index_t linear_idx, MultiIndex& result)
    {
        // Compute ordered indices and assign to result in original dimension order
        ((result(Number<New2Old::At(Number<Is>{})>{}) = (linear_idx / strides[Is]) % lengths[Is]),
         ...);
    }
};

} // namespace detail

/**
 * @brief Compile-time N-dimensional loop with static multi-indices.
 *
 * Iterates over an N-dimensional space where dimensions have sizes specified
 * by Lengths. The iteration order is controlled by Orders. Each iteration
 * provides a compile-time Sequence containing the current multi-index.
 *
 * Uses O(1) template instantiation depth via flat loop with index decomposition,
 * avoiding recursive template structures.
 *
 * Example:
 * @code
 * // Iterate over 2x3 space in row-major order (dim 0 outer, dim 1 inner)
 * static_ford<Sequence<2, 3>>{}([](auto multi_id) {
 *     constexpr index_t i = multi_id[Number<0>{}];  // 0, 0, 0, 1, 1, 1
 *     constexpr index_t j = multi_id[Number<1>{}];  // 0, 1, 2, 0, 1, 2
 * });
 *
 * // Column-major order (dim 1 outer, dim 0 inner)
 * static_ford<Sequence<2, 3>, Sequence<1, 0>>{}([](auto multi_id) {
 *     // Visits: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
 * });
 * @endcode
 *
 * @tparam Lengths Sequence<L0, L1, ...> specifying dimension sizes
 * @tparam Orders Sequence<O0, O1, ...> specifying iteration order
 *                (default: Sequence<0, 1, ..., N-1> for row-major)
 */
template <class Lengths, class Orders = make_index_sequence<Lengths::GetSize()>>
struct static_ford : detail::ford_base<Lengths, Orders>
{
    using Base       = detail::ford_base<Lengths, Orders>;
    using Decomposer = detail::index_decomposer<typename Base::OrderedLengthsType,
                                                make_index_sequence<Base::NDim>>;

    /**
     * @brief Execute the N-dimensional loop.
     *
     * Calls f with a compile-time Sequence<i0, i1, ...> for each point
     * in the iteration space.
     *
     * @tparam F Functor type with signature F(Sequence<...>)
     * @param f The functor to call for each multi-index
     */
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        static_for<0, Base::TotalSize, 1>{}([&](auto linear_idx) {
            using OrderedIdx = typename Decomposer::template decompose<linear_idx.value>;
            f(OrderedIdx::ReorderGivenOld2New(Orders{}));
        });
    }
};

/**
 * @brief Runtime N-dimensional loop with runtime multi-indices.
 *
 * Iterates over an N-dimensional space where dimensions have sizes specified
 * by Lengths. The iteration order is controlled by Orders. Each iteration
 * provides a runtime multi-index container.
 *
 * Uses O(1) template instantiation depth via flat for-loop with index decomposition,
 * avoiding recursive template structures.
 *
 * Example:
 * @code
 * // Iterate over 2x3 space in row-major order
 * ford<Sequence<2, 3>>{}([](auto multi_id) {
 *     index_t i = multi_id[Number<0>{}];  // Runtime values
 *     index_t j = multi_id[Number<1>{}];
 * });
 * @endcode
 *
 * @tparam Lengths Sequence<L0, L1, ...> specifying dimension sizes
 * @tparam Orders Sequence<O0, O1, ...> specifying iteration order
 *                (default: Sequence<0, 1, ..., N-1> for row-major)
 */
template <class Lengths, class Orders = make_index_sequence<Lengths::GetSize()>>
struct ford : detail::ford_base<Lengths, Orders>
{
    using Base       = detail::ford_base<Lengths, Orders>;
    using Decomposer = detail::index_decomposer<typename Base::OrderedLengthsType,
                                                make_index_sequence<Base::NDim>>;

    /**
     * @brief Execute the N-dimensional loop.
     *
     * Calls f with a runtime multi-index for each point in the iteration space.
     *
     * @tparam F Functor type with signature F(MultiIndex)
     * @param f The functor to call for each multi-index
     */
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        for(index_t linear_idx = 0; linear_idx < Base::TotalSize; ++linear_idx)
        {
            auto multi_id = make_zero_multi_index<Base::NDim>();
            Decomposer::template decompose_runtime<Orders>(linear_idx, multi_id);
            f(multi_id);
        }
    }
};

} // namespace ck
