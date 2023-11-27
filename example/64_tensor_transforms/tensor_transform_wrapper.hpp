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
namespace tensor_transform_wrapper {

/**
 * \brief Layout wrapper
 *
 * \details
 * Layout wrapper that performs the tensor descriptor logic.
 *
 * \tparam Shape Tuple of Number<> (for compile-time layout) or index_t
 *         (dynamic layout). It is possible to pass nested shapes
 *         (e.g. ((4, 2), 2)), nested dimensions are merged.
 * \tparam Strides Tuple of Number<> (for compile-time layout) or index_t
 *         (dynamic layout). Stride tuple should be nested if shape tuple is
 *         nested.
 */
template <typename Shape, typename Strides = Tuple<>>
struct Layout
{
    private:
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    template <typename T>
    using is_tuple = decltype(std::declval<T&>().IsTuple());

    // Generate packed (column-major) strides if not passed
    template <typename... Ts>
    __host__ __device__ constexpr static auto
    GenerateColumnMajorPackedStrides(const Tuple<Ts...>& tuple)
    {
        return generate_tuple(
            [&](auto i) {
                if constexpr(i.value == 0)
                {
                    return I1;
                }
                else
                {
                    return TupleReduce<I0.value, i.value>([](auto x, auto y) { return x * y; },
                                                          tuple);
                }
            },
            Number<Tuple<Ts...>::Size()>{});
    }

    template <typename Idx, typename... Ts>
    __host__ __device__ constexpr static auto GenerateLowerDim(const Tuple<Ts...>& tuple)
    {
        if constexpr(Idx::value == 0)
        {
            if constexpr(is_detected<is_tuple, tuple_element_t<Idx::value, Tuple<Ts...>>>::value)
            {
                constexpr index_t merge_nelems =
                    decltype(UnrollNestedTuple(tuple.At(Idx{})))::Size();
                using LowerDimsSequence =
                    typename arithmetic_sequence_gen<0, merge_nelems, 1>::type;
                return LowerDimsSequence::Reverse();
            }
            else
            {
                return Sequence<0>{};
            }
        }
        else
        {
            using PreviousSeqT      = decltype(GenerateLowerDim<Number<Idx::value - 1>>(tuple));
            const auto next_seq_val = PreviousSeqT::At(I0) + 1;
            if constexpr(is_detected<is_tuple, tuple_element_t<Idx::value, Tuple<Ts...>>>::value)
            {
                constexpr index_t merge_nelems =
                    decltype(UnrollNestedTuple(tuple.At(Idx{})))::Size();
                using LowerDimsSequence =
                    typename arithmetic_sequence_gen<next_seq_val, next_seq_val + merge_nelems, 1>::
                        type;
                return LowerDimsSequence::Reverse();
            }
            else
            {
                return Sequence<next_seq_val>{};
            }
        }
    }

    template <typename... ShapeDims, typename... IdxDims>
    __host__ __device__ constexpr static auto UnrollShapeViaIdx(const Tuple<ShapeDims...>& shape,
                                                                const Tuple<IdxDims...>& idx)
    {
        if constexpr(!IsTupleNested(Tuple<IdxDims...>{}))
        {
            // Index unrolled to flatten, return shape
            return shape;
        }
        else
        {
            // Iterate over shape tuple elements:
            // 1. If coressponding idx element is tuple then return (will be unrolled)
            // 2. If no, pack in tuple. It will be restored during unroll.
            auto unrolled_shape_via_idx = generate_tuple(
                [&](auto i) {
                    if constexpr(is_detected<is_tuple,
                                             tuple_element_t<i, Tuple<IdxDims...>>>::value)
                    {
                        return shape.At(i);
                    }
                    else
                    {
                        return make_tuple(shape.At(i));
                    }
                },
                Number<Tuple<IdxDims...>::Size()>{});

            // Unroll and process next step
            return UnrollShapeViaIdx(UnrollNestedTuple<0, 1>(unrolled_shape_via_idx),
                                     UnrollNestedTuple<0, 1>(idx));
        }
    }

    template <typename... ShapeDims, typename DescriptorToMerge>
    __host__ __device__ constexpr static auto MakeMerge1d(const Tuple<ShapeDims...>& shape,
                                                          DescriptorToMerge& desc)
    {
        // Reverse each element in tuple
        using ReversedUnrolledShape = decltype(ReverseTuple(UnrollNestedTuple(shape)));
        const auto merge_elems      = ReversedUnrolledShape{};

        // Generate reverted indexes (column major traverse)
        using MergeElemsSequence =
            typename arithmetic_sequence_gen<0, ReversedUnrolledShape::Size(), 1>::type;
        const auto lower_dims = make_tuple(MergeElemsSequence::Reverse());
        const auto upper_dims = make_tuple(Sequence<0>{});
        // Merge to 1d
        return transform_tensor_descriptor(
            desc, make_tuple(make_merge_transform(merge_elems)), lower_dims, upper_dims);
    }

    template <typename... ShapeDims, typename... IdxDims, typename DescriptorToMerge>
    __host__ __device__ constexpr static auto
    MakeMerges(const Tuple<ShapeDims...>& shape, const Tuple<IdxDims...>&, DescriptorToMerge& desc)
    {
        const auto transforms = generate_tuple(
            [&](auto i) {
                // Compare Idx with shape
                if constexpr(is_detected<is_tuple,
                                         tuple_element_t<i, Tuple<ShapeDims...>>>::value &&
                             !is_detected<is_tuple, tuple_element_t<i, Tuple<IdxDims...>>>::value)
                {
                    // If shape element is tuple and idx element is Number, then merge
                    // Unroll and reverse tuple to traverse column-major
                    const auto merge_elems = ReverseTuple(UnrollNestedTuple(shape.At(i)));
                    return make_merge_transform(merge_elems);
                }
                else
                {
                    // If shape element is integer and idx element is tuple, passed idx is wrong
                    static_assert(
                        !(!is_detected<is_tuple, tuple_element_t<i, Tuple<ShapeDims...>>>::value &&
                          is_detected<is_tuple, tuple_element_t<i, Tuple<IdxDims...>>>::value),
                        "Wrong Idx for layout()");
                    // If shape element has the same type as idx element, then pass through
                    return make_pass_through_transform(shape.At(i));
                }
            },
            Number<Tuple<ShapeDims...>::Size()>{});

        const auto lower_dims =
            generate_tuple([&](auto i) { return GenerateLowerDim<Number<i>>(shape); },
                           Number<Tuple<ShapeDims...>::Size()>{});
        const auto upper_dims = generate_tuple([&](auto i) { return Sequence<i.value>{}; },
                                               Number<Tuple<ShapeDims...>::Size()>{});

        return transform_tensor_descriptor(desc, transforms, lower_dims, upper_dims);
    }

    template <typename... ShapeDims, typename... IdxDims>
    __host__ __device__ constexpr auto TransformDesc(const Tuple<ShapeDims...>& shape,
                                                     const Tuple<IdxDims...>& idx) const
    {
        if constexpr(Tuple<IdxDims...>::Size() == I1)
        {
            // 1d idx path
            return MakeMerge1d(shape, descriptor_);
        }
        else
        {
            static_assert(Tuple<ShapeDims...>::Size() == Tuple<IdxDims...>::Size(),
                          "Idx rank and Shape rank must be the same (except 1d).");
            // Unroll while IdxDims is nested
            const auto unrolled_shape_via_idx = UnrollShapeViaIdx(shape, idx);
            // Transform correct form of shape
            return MakeMerges(unrolled_shape_via_idx, UnrollNestedTuple(idx), descriptor_);
        }
    }

    template <typename LayoutShape, typename LayoutStrides>
    __host__ __device__ static auto MakeNaiveDescriptor(const LayoutShape& shape,
                                                        const LayoutStrides& strides)
    {
        const auto unrolled_shape = UnrollNestedTuple(shape);

        if constexpr(ck::is_same_v<LayoutStrides, Tuple<>>)
        {
            // If shape is packed
            const auto column_major_packed_strides =
                GenerateColumnMajorPackedStrides(unrolled_shape);
            return make_naive_tensor_descriptor(unrolled_shape, column_major_packed_strides);
        }
        else
        {
            const auto unrolled_strides = UnrollNestedTuple(strides);
            static_assert(unrolled_shape.Size() == unrolled_strides.Size(),
                          "Size of strides and shape are not consistent.");
            return make_naive_tensor_descriptor(unrolled_shape, unrolled_strides);
        }
    }

    public:
    using NaiveDescriptorType = remove_cvref_t<decltype(MakeNaiveDescriptor(Shape{}, Strides{}))>;

    /**
     * \brief Layout constructor.
     *
     * \param shape Shape for layout.
     * \param strides Strides for layout (optional if tensor is packed).
     * \return Layout object.
     */
    __host__ __device__ Layout() = delete;
    __host__ __device__ Layout(const Shape& shape, const Strides& strides) : descriptor_{}
    {
        // Construct if runtime mode
        if constexpr(!NaiveDescriptorType::IsKnownAtCompileTime())
        {
            // Keep only shape, strides are not need for transforms
            shape_      = shape;
            descriptor_ = MakeNaiveDescriptor(shape, strides);
        }
    }

    __host__ __device__ Layout(const Shape& shape) : descriptor_{}
    {
        if constexpr(!NaiveDescriptorType::IsKnownAtCompileTime())
        {
            shape_      = shape;
            descriptor_ = MakeNaiveDescriptor(shape, Strides{});
        }
    }

    /**
     * \brief Returns real offset to element as const in runtime.
     *
     * \tparam Idxs Tuple of indexes.
     * \return Calculated offset as const.
     */
    template <typename Idxs>
    __host__ __device__ constexpr index_t operator()() const
    {
        using TransformedDesc = decltype(TransformDesc(Shape{}, Idxs{}));
        using UnrolledIdx     = decltype(UnrollNestedTuple(Idxs{}));
        return TransformedDesc{}.CalculateOffset(UnrolledIdx{});
    }

    /**
     * \brief Returns real offset to element in runtime.
     *
     * \tparam Idxs Tuple of indexes.
     * \return Calculated offset.
     */
    template <typename Idxs>
    __host__ __device__ constexpr index_t operator()()
    {
        using TransformedDesc = decltype(TransformDesc(Shape{}, Idxs{}));
        using UnrolledIdx     = decltype(UnrollNestedTuple(Idxs{}));
        return TransformedDesc{}.CalculateOffset(UnrolledIdx{});
    }

    /**
     * \brief Returns real offset to element in compile time.
     *
     * \param Idx Tuple of indexes.
     * \return Calculated offset.
     */
    template <typename... Ts>
    __host__ __device__ index_t operator()(const Tuple<Ts...>& Idx) const
    {
        // Static to construct transformed_desc only once
        static const auto transformed_desc = TransformDesc(shape_, Idx);
        return transformed_desc.CalculateOffset(UnrollNestedTuple(Idx));
    }

    /**
     * \brief Length getter (product if tuple) as const.
     *
     * \tparam IDim Tuple of indexes or index.
     * \return Calculated size.
     */
    template <index_t IDim>
    __host__ __device__ constexpr index_t GetLength() const
    {
        const auto elem = shape_.At(Number<IDim>{});
        if constexpr(is_detected<is_tuple, tuple_element_t<IDim, Shape>>::value)
        {
            const auto unrolled_element = UnrollNestedTuple(elem);
            return TupleReduce<I0.value, unrolled_element.Size()>(
                [](auto x, auto y) { return x * y; }, unrolled_element);
        }
        else
        {
            return elem;
        }
    }

    /**
     * \brief Length getter (product if tuple).
     *
     * \tparam IDim Tuple of indexes or index.
     * \return Calculated size.
     */
    template <index_t IDim>
    __host__ __device__ constexpr index_t GetLength()
    {
        const auto elem = shape_.At(Number<IDim>{});
        if constexpr(is_detected<is_tuple, tuple_element_t<IDim, Shape>>::value)
        {
            const auto unrolled_element = UnrollNestedTuple(elem);
            return TupleReduce<I0.value, unrolled_element.Size()>(
                [](auto x, auto y) { return x * y; }, unrolled_element);
        }
        else
        {
            return elem;
        }
    }

    /**
     * \brief Layout size getter (product of shape) as const.
     *
     * \return Calculated size.
     */
    __host__ __device__ constexpr index_t GetLength() const
    {
        const auto unrolled_shape = UnrollNestedTuple(shape_);
        return TupleReduce<I0.value, unrolled_shape.Size()>([](auto x, auto y) { return x * y; },
                                                            unrolled_shape);
    }

    /**
     * \brief Layout size getter (product of shape).
     *
     * \return Calculated size.
     */
    __host__ __device__ constexpr index_t GetLength()
    {
        const auto unrolled_shape = UnrollNestedTuple(shape_);
        return TupleReduce<I0.value, unrolled_shape.Size()>([](auto x, auto y) { return x * y; },
                                                            unrolled_shape);
    }

    /**
     * \brief Dimension getter as const.
     *
     * \tparam IDim Dimension idx.
     * \return Calculated size.
     */
    template <index_t IDim>
    __host__ __device__ constexpr auto Get() const
    {
        const auto elem = shape_.At(Number<IDim>{});
        return elem;
    }

    /**
     * \brief Dimension getter.
     *
     * \tparam IDim Dimension idx.
     * \return Calculated size.
     */
    template <index_t IDim>
    __host__ __device__ constexpr auto Get()
    {
        const auto elem = shape_.At(Number<IDim>{});
        return elem;
    }

    private:
    NaiveDescriptorType descriptor_;
    Shape shape_;
};

// Layout helpers
// Length getter (product if tuple)
template <index_t idx, typename Shape, typename Strides>
__host__ __device__ constexpr index_t size(const Layout<Shape, Strides>& layout)
{
    return layout.template GetLength<idx>();
}

// Get shape size (product of dims if tuple)
template <typename... ShapeDims>
__host__ __device__ constexpr index_t size(const Tuple<ShapeDims...>& shape)
{
    using UnrolledShape = decltype(UnrollNestedTuple(shape));
    return TupleReduce<0, UnrolledShape::Size()>([](auto x, auto y) { return x * y; },
                                                 UnrolledShape{});
}

// Get dim size (could be returned from get function)
template <typename T>
__host__ __device__ T constexpr size(const T& dim)
{
    return dim;
}

// Get layout size (product of shapes)
template <typename Shape, typename Strides>
__host__ __device__ constexpr index_t size(const Layout<Shape, Strides>& layout)
{
    return layout.GetLength();
}

// Get shape element size
template <index_t idx, typename... ShapeDims>
__host__ __device__ constexpr index_t size(const Tuple<ShapeDims...>& shape)
{
    return size(shape.At(Number<idx>{}));
}

// Dim getter (tuple if tuple)
template <index_t idx, typename Shape, typename Strides>
__host__ __device__ constexpr auto get(const Layout<Shape, Strides>& layout)
{
    return layout.template Get<idx>();
}

template <typename Shape, typename Strides>
__host__ __device__ constexpr Layout<Shape, Strides> make_layout(const Shape& shape,
                                                                 const Strides& strides)
{
    return Layout<Shape, Strides>(shape, strides);
}

template <typename Shape>
__host__ __device__ constexpr Layout<Shape> make_layout(const Shape& shape)
{
    return Layout<Shape>(shape);
}

} // namespace tensor_transform_wrapper
} // namespace ck
