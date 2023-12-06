// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/tuple_helper.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/is_detected.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"

namespace ck {
namespace wrapper {

// Disable from doxygen docs generation
/// @cond
// forward declaration
template <typename Shape, typename Strides = Tuple<>>
struct Layout;

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());
/// @endcond

// make_*
/**
 * \brief Make layout function.
 *
 * \tparam Shape Shape for layout.
 * \tparam Strides Strides for layout.
 * \return Constructed layout.
 */
template <typename Shape, typename Strides>
__host__ __device__ constexpr Layout<Shape, Strides> make_layout(const Shape& shape,
                                                                 const Strides& strides)
{
    return Layout<Shape, Strides>(shape, strides);
}

/**
 * \brief Make layout function with packed strides
 *        (column-major).
 *
 * \tparam Shape Shape for layout.
 * \return Constructed layout.
 */
template <typename Shape>
__host__ __device__ constexpr Layout<Shape> make_layout(const Shape& shape)
{
    return Layout<Shape>(shape);
}

// Layout helpers
// get
/**
 * \brief Get element from tuple (Shape/Strides/Idxs).
 *
 * \tparam idx Index to lookup.
 * \param tuple Tuple to lookup.
 * \return Requsted element.
 */
template <index_t idx, typename... Dims>
__host__ __device__ constexpr auto get(const Tuple<Dims...>& tuple)
{
    return tuple.At(Number<idx>{});
}

/**
 * \brief Get sub layout.
 *
 * \tparam idx Index to lookup.
 * \param layout Layout to create sub layout.
 * \return Requsted sub layout.
 */
template <index_t idx, typename Shape, typename Strides>
__host__ __device__ constexpr auto get(const Layout<Shape, Strides>& layout)
{
    const auto new_shape = get<idx>(layout.GetShape());
    static_assert(is_detected<is_tuple, decltype(new_shape)>::value,
                  "Shape of sub layout must be tuple");
    if constexpr(is_same_v<Strides, Tuple<>>)
    {
        // If stride not passed, create without strides
        return make_layout(new_shape);
    }
    else
    {
        const auto new_strides = get<idx>(layout.GetStrides());
        static_assert(is_detected<is_tuple, decltype(new_strides)>::value,
                      "Strides of sub layout must be tuple");
        return make_layout(new_shape, new_strides);
    }
}

/**
 * \brief Hierarchical get.
 *
 * \tparam Idxs Indexes to lookup.
 * \param elem Element to lookup.
 * \return Requsted element.
 */
template <index_t Idx, index_t... Idxs, typename T>
__host__ __device__ constexpr auto get(const T& elem)
{
    return get<Idxs...>(get<Idx>(elem));
}

// size
/**
 * \brief Length get (product if tuple).
 *
 * \tparam idx Index to lookup.
 * \param layout Layout to get Shape.
 * \return Requsted length.
 */
template <index_t idx, typename Shape, typename Strides>
__host__ __device__ constexpr index_t size(const Layout<Shape, Strides>& layout)
{
    return layout.template GetLength<idx>();
}

/**
 * \brief Shape size (product of dims).
 *
 * \param shape Shape to lookup.
 * \return Requsted size.
 */
template <typename... ShapeDims>
__host__ __device__ constexpr index_t size(const Tuple<ShapeDims...>& shape)
{
    const auto unrolled_shape = UnrollNestedTuple(shape);
    return TupleReduce<0, unrolled_shape.Size()>([](auto x, auto y) { return x * y; },
                                                 unrolled_shape);
}

// Get dim size (could be returned from get function)
/**
 * \private
 */
template <typename T>
__host__ __device__ T constexpr size(const T& dim)
{
    return dim;
}

/**
 * \brief Layout size (product of dims).
 *
 * \param layout Layout to calculate shape size.
 * \return Requsted size.
 */
template <typename Shape, typename Strides>
__host__ __device__ constexpr index_t size(const Layout<Shape, Strides>& layout)
{
    return layout.GetLengths();
}

/**
 * \brief Length get from tuple (product if tuple).
 *
 * \tparam idx Index to lookup.
 * \param tuple Tuple to lookup.
 * \return Requsted length.
 */
template <index_t idx, typename... Ts>
__host__ __device__ constexpr index_t size(const Tuple<Ts...>& tuple)
{
    return size(tuple.At(Number<idx>{}));
}

/**
 * \brief Hierarchical size.
 *
 * \tparam Idxs Indexes to lookup.
 * \param elem Element to lookup.
 * \return Requsted element.
 */
template <index_t... Idxs, typename T>
__host__ __device__ constexpr auto size(const T& elem)
{
    return size(get<Idxs...>(elem));
}

// rank
/**
 * \brief Get layout rank (num elements in shape).
 *
 * \param layout Layout to calculate rank.
 * \return Requsted rank.
 */
template <typename Shape, typename Strides>
__host__ __device__ constexpr auto rank([[maybe_unused]] const Layout<Shape, Strides>& layout)
{
    return Shape::Size();
}

/**
 * \brief Get tuple rank (num elements in tuple).
 *        Return 1 if scalar passed.
 *
 * \param tuple Tuple to calculate rank.
 * \return Requsted rank.
 */
template <typename... Dims>
__host__ __device__ constexpr auto rank([[maybe_unused]] const Tuple<Dims...>& tuple)
{
    return Tuple<Dims...>::Size();
}

/**
 * \private
 */
template <index_t IDim>
__host__ __device__ constexpr index_t rank(const Number<IDim>&)
{
    return 1;
}

/**
 * \private
 */
__host__ __device__ constexpr index_t rank(const index_t&) { return 1; }

/**
 * \brief Hierarchical rank.
 *
 * \tparam Idxs Indexes to lookup.
 * \param elem Element to lookup.
 * \return Requsted rank.
 */
template <index_t... Idxs, typename T>
__host__ __device__ constexpr auto rank(const T& elem)
{
    return rank(get<Idxs...>(elem));
}

// depth
/**
 * \brief Get depth of the layout shape (return 0 if scalar).
 *
 * \param layout Layout to calculate depth.
 * \return Requsted depth.
 */
template <typename Shape, typename Strides>
__host__ __device__ constexpr auto depth(const Layout<Shape, Strides>& layout)
{
    return TupleDepth(layout.GetShape());
}

/**
 * \brief Get depth of the tuple. (return 0 if scalar)
 *
 * \param tuple Tuple to calculate depth.
 * \return Requsted depth.
 */
template <typename... Dims>
__host__ __device__ constexpr auto depth(const Tuple<Dims...>& tuple)
{
    return TupleDepth(tuple);
}

/**
 * \private
 */
template <index_t IDim>
__host__ __device__ constexpr index_t depth(const Number<IDim>&)
{
    return 0;
}

/**
 * \private
 */
__host__ __device__ constexpr index_t depth(const index_t&) { return 0; }

/**
 * \brief Hierarchical depth.
 *
 * \tparam Idxs Indexes to lookup.
 * \param elem Element to lookup.
 * \return Requsted depth.
 */
template <index_t... Idxs, typename T>
__host__ __device__ constexpr auto depth(const T& elem)
{
    return depth(get<Idxs...>(elem));
}

/**
 * \brief Get Layout strides.
 *
 * \param layout Layout to get strides.
 * \return Requsted strides.
 */
template <typename Shape, typename Strides>
__host__ __device__ constexpr auto stride(const Layout<Shape, Strides>& layout)
{
    return layout.GetStrides();
}

/**
 * \brief Get Layout shape.
 *
 * \param layout Layout to get shape.
 * \return Requsted shape.
 */
template <typename Shape, typename Strides>
__host__ __device__ constexpr auto shape(const Layout<Shape, Strides>& layout)
{
    return layout.GetShape();
}

} // namespace wrapper
} // namespace ck
