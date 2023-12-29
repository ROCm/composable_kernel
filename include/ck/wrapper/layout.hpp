// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/wrapper/utils/layout_utils.hpp"

namespace ck {
namespace wrapper {

/**
 * \brief Layout wrapper that performs the tensor descriptor logic.
 *
 * \tparam Shape Tuple of Number<> (for compile-time layout) or index_t
 *         (dynamic layout). It is possible to pass nested shapes
 *         (e.g. ((4, 2), 2)), nested dimensions are merged.
 * \tparam UnnestedDescriptorType Tensor descriptor for unnested shape dims.
 */
template <typename Shape, typename UnnestedDescriptorType>
struct Layout
{
    private:
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    // Generate default idxs tuple (idx with all merged nested shapes)
    template <typename... Ts>
    __host__ __device__ constexpr static auto GenerateDefaultIdxsTuple(const Tuple<Ts...>&)
    {
        return generate_tuple(
            [&](auto) {
                if constexpr(!UnnestedDescriptorType::IsKnownAtCompileTime())
                {
                    // runtime layout
                    return index_t(0);
                }
                else
                {
                    // compiletime layout
                    return I0;
                }
            },
            Number<Tuple<Ts...>::Size()>{});
    }

    // Generate LowerDims in Compile-time for MergeTrasform using passed Type
    // If element of Tuple<Ts...> is also tuple, then merge (generate sequence for merge)
    // If tuple is element, then pass through (sequence with one element)
    template <typename Idx, typename... Ts>
    __host__ __device__ constexpr static auto GenerateLowerDim(const Tuple<Ts...>&)
    {
        if constexpr(Idx::value == 0)
        {
            if constexpr(is_detected<is_tuple, tuple_element_t<Idx::value, Tuple<Ts...>>>::value)
            {
                // Return Sequence for the first tuple
                constexpr index_t merge_nelems = decltype(UnrollNestedTuple(
                    tuple_element_t<Idx::value, Tuple<Ts...>>{}))::Size();
                using LowerDimsSequence =
                    typename arithmetic_sequence_gen<0, merge_nelems, 1>::type;
                return LowerDimsSequence::Reverse();
            }
            else
            {
                // Return first element
                return Sequence<0>{};
            }
        }
        else
        {
            // Get previous element using recurence (in compile-time)
            using PreviousSeqT = decltype(GenerateLowerDim<Number<Idx::value - 1>>(Tuple<Ts...>{}));
            const auto next_seq_val = PreviousSeqT::At(I0) + 1;
            if constexpr(is_detected<is_tuple, tuple_element_t<Idx::value, Tuple<Ts...>>>::value)
            {
                constexpr index_t merge_nelems = decltype(UnrollNestedTuple(
                    tuple_element_t<Idx::value, Tuple<Ts...>>{}))::Size();
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

    // Iterate over nested tuples in shape
    // Unroll nested tuples to align Tuple<ShapeDims...> to Tuple<IdxDims...>
    // Example idx:     (1,      1), 1,      1
    // Example shape:   (2, (2, 2)), 2, (2, 2)
    // Unrolled shape:  2,  (2, 2),  2, (2, 2)
    template <typename... ShapeDims, typename... IdxDims>
    __host__ __device__ constexpr static auto AlignShapeToIdx(const Tuple<ShapeDims...>& shape,
                                                              const Tuple<IdxDims...>& idx)
    {
        if constexpr(!IsNestedTuple(Tuple<IdxDims...>{}))
        {
            // Index unrolled to flatten, return shape
            return shape;
        }
        else
        {
            // Iterate over shape tuple elements:
            // 1. If corresponding idx element is tuple then return (will be unrolled)
            // 2. If no, pack in tuple. It will be restored during unroll.
            auto aligned_shape = generate_tuple(
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
            return AlignShapeToIdx(UnrollNestedTuple<0, 1>(aligned_shape),
                                   UnrollNestedTuple<0, 1>(idx));
        }
    }

    template <typename... ShapeDims, typename DescriptorToMerge>
    __host__ __device__ constexpr static auto MakeMerge1d(const Tuple<ShapeDims...>& shape,
                                                          const DescriptorToMerge& desc)
    {
        // Reverse each element in tuple
        const auto merge_elems = TupleReverse(UnrollNestedTuple(shape));
        // Generate reverted indexes (column major traverse)
        using MergeElemsSequence = typename arithmetic_sequence_gen<0, merge_elems.Size(), 1>::type;
        const auto lower_dims    = make_tuple(MergeElemsSequence::Reverse());
        const auto upper_dims    = make_tuple(Sequence<0>{});
        // Merge to 1d
        return transform_tensor_descriptor(
            desc, make_tuple(make_merge_transform(merge_elems)), lower_dims, upper_dims);
    }

    // Merge nested shape dims when corresponding index is also nested.
    // Input desc shape: 2,  2,  2, 2,  2,  2
    // Example idx:      1,      1, 1,      1
    // Example shape:    2, (2, 2), 2, (2, 2)
    // Merged shape:     2,      4, 2,      4
    template <typename... ShapeDims, typename... IdxDims, typename DescriptorToMerge>
    __host__ __device__ constexpr static auto CreateMergedDescriptor(
        const Tuple<ShapeDims...>& shape, const Tuple<IdxDims...>&, DescriptorToMerge& desc)
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
                    const auto merge_elems = TupleReverse(UnrollNestedTuple(shape.At(i)));
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

    using Descriptor1dType =
        remove_cvref_t<decltype(MakeMerge1d(Shape{}, UnnestedDescriptorType{}))>;
    using DefaultIdxsTupleType = remove_cvref_t<decltype(GenerateDefaultIdxsTuple(Shape{}))>;

    template <typename... ShapeDims, typename... IdxDims>
    __host__ __device__ constexpr static auto
    TransformDesc(const Tuple<ShapeDims...>& shape,
                  const Tuple<IdxDims...>& idx,
                  const UnnestedDescriptorType& naive_descriptor)
    {
        if constexpr(Tuple<IdxDims...>::Size() == I1)
        {
            // 1d idx path
            return MakeMerge1d(shape, naive_descriptor);
        }
        else
        {
            // Merge nested shape dims
            // Example idx:   (1,      1), 1,      1
            // Example shape: (2, (2, 2)), 2, (2, 2)
            // Merged shape:  (2,      4), 2,      4
            static_assert(Tuple<ShapeDims...>::Size() == Tuple<IdxDims...>::Size(),
                          "Idx rank and Shape rank must be the same (except 1d).");
            // Unroll while IdxDims is nested
            const auto aligned_shape = AlignShapeToIdx(shape, idx);
            // Transform correct form of shape
            return CreateMergedDescriptor(aligned_shape, UnrollNestedTuple(idx), naive_descriptor);
        }
    }

    using MergedNestsDescriptorType = remove_cvref_t<decltype(TransformDesc(
        Shape{}, DefaultIdxsTupleType{}, UnnestedDescriptorType{}))>;

    public:
    __host__ __device__ constexpr auto GetElementSpaceSize() const
    {
        return unnested_descriptor_.GetElementSpaceSize();
    }

    __host__ __device__ Layout() = delete;

    /**
     * \brief Layout constructor.
     *
     * \param shape Shape for layout.
     * \param unnested_descriptor Descriptor
     */
    __host__ __device__ constexpr Layout(const Shape& shape,
                                         const UnnestedDescriptorType& unnested_descriptor)
        : shape_(shape)
    {
        // Construct if runtime mode
        if constexpr(!UnnestedDescriptorType::IsKnownAtCompileTime())
        {
            unnested_descriptor_ = unnested_descriptor;
            descriptor_1d_       = MakeMerge1d(shape_, unnested_descriptor_);
            merged_nests_descriptor_ =
                TransformDesc(shape_, DefaultIdxsTupleType{}, unnested_descriptor_);
        }
    }

    /**
     * \brief Returns real offset to element in runtime.
     *
     * \tparam Idxs Tuple of indexes.
     * \return Calculated offset.
     */
    template <typename Idxs>
    __host__ __device__ constexpr index_t operator()() const
    {
        static_assert(UnnestedDescriptorType::IsKnownAtCompileTime(),
                      "Compiletime operator used on runtime layout.");
        using TransformedDesc = decltype(TransformDesc(Shape{}, Idxs{}, UnnestedDescriptorType{}));
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
        if constexpr(!IsNestedTuple(Tuple<Ts...>{}) && Tuple<Ts...>::Size() == 1)
        {
            // if 1d access
            return descriptor_1d_.CalculateOffset(Idx);
        }
        else if constexpr(!IsNestedTuple(Tuple<Ts...>{}) && Tuple<Ts...>::Size() == Shape::Size())
        {
            // if Shape::Size() access (merged nested shapes)
            return merged_nests_descriptor_.CalculateOffset(UnrollNestedTuple(Idx));
        }
        else
        {
            // Custom index, need to transform descriptor
            const auto transformed_desc = TransformDesc(shape_, Idx, unnested_descriptor_);
            return transformed_desc.CalculateOffset(UnrollNestedTuple(Idx));
        }
    }

    /**
     * \brief Length getter (product if tuple).
     *
     * \tparam IDim Tuple of indexes or index.
     * \return Calculated size.
     */
    template <index_t IDim>
    __host__ __device__ constexpr auto GetLength() const
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
     * \brief Layout size getter (product of shape).
     *
     * \return Calculated size.
     */
    __host__ __device__ constexpr auto GetLengths() const
    {
        const auto unrolled_shape = UnrollNestedTuple(shape_);
        return TupleReduce<I0.value, unrolled_shape.Size()>([](auto x, auto y) { return x * y; },
                                                            unrolled_shape);
    }

    /**
     * \brief Shape getter.
     *
     * \return Shape.
     */
    __host__ __device__ constexpr const Shape& GetShape() const { return shape_; }

    /**
     * \brief Get default lengths (tuple filled with Shape length elements).
     *
     * \return Default lengths.
     */
    __host__ __device__ constexpr auto GetDefaultLengthsTuple() const
    {
        return generate_tuple([&](auto i) { return GetLength<i>(); }, Number<Shape::Size()>{});
    }

    /**
     * \brief Get default start idx (tuple filled with 0s of the same size as Shape).
     *
     * \return Default start idx.
     */
    __host__ __device__ constexpr auto GetDefaultStartIdxs() const
    {
        return GenerateDefaultIdxsTuple(shape_);
    }

    /**
     * \brief Get default descriptor (with the same size as Shape)
     *
     * \return Default descriptor.
     */
    __host__ __device__ constexpr const MergedNestsDescriptorType& GetDefaultDescriptor() const
    {
        return merged_nests_descriptor_;
    }

    /**
     * \brief Get unnested descriptor (with unrolled dims)
     *
     * \return Flatten descriptor.
     */
    __host__ __device__ constexpr const UnnestedDescriptorType& GetUnnestedDescriptor() const
    {
        return unnested_descriptor_;
    }

    private:
    UnnestedDescriptorType unnested_descriptor_;
    Descriptor1dType descriptor_1d_;
    MergedNestsDescriptorType merged_nests_descriptor_;
    const Shape shape_;
};

} // namespace wrapper
} // namespace ck
