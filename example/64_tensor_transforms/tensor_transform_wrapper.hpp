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
    template <typename T>
    using is_tuple = decltype(std::declval<T&>().IsTuple());

    template <typename Tuple, typename Idx>
    constexpr static auto GenerateLowerDim(Tuple tuple)
    {
        if constexpr(Idx::value == 0)
        {
            if constexpr(is_detected<is_tuple, tuple_element_t<Idx::value, Tuple>>::value)
            {
                constexpr index_t merge_nelems =
                    decltype(UnrollNestedTuple(tuple.At(Idx{})))::Size();
                return typename arithmetic_sequence_gen<0, merge_nelems, 1>::type{};
            }
            else
            {
                return Sequence<0>{};
            }
        }
        else
        {
            using PreviousSeqT = decltype(GenerateLowerDim<Tuple, Number<Idx::value - 1>>(tuple));
            const auto next_seq_val = PreviousSeqT::At(PreviousSeqT::Size() - 1) + 1;
            if constexpr(is_detected<is_tuple, tuple_element_t<Idx::value, Tuple>>::value)
            {
                constexpr index_t merge_nelems =
                    decltype(UnrollNestedTuple(tuple.At(Idx{})))::Size();
                return typename arithmetic_sequence_gen<next_seq_val,
                                                        next_seq_val + merge_nelems,
                                                        1>::type{};
            }
            else
            {
                return Sequence<next_seq_val>{};
            }
        }
    }

    template <typename Tuple, typename Descriptor>
    constexpr static auto MakeMerges(const Tuple& tuple, Descriptor& desc)
    {
        const auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(is_detected<is_tuple, tuple_element_t<i, Tuple>>::value)
                {
                    const auto merge_elems = UnrollNestedTuple(tuple.At(i));
                    return make_merge_transform(merge_elems);
                }
                else
                {
                    return make_pass_through_transform(tuple.At(i));
                }
            },
            Number<Tuple::Size()>{});

        const auto lower_dims =
            generate_tuple([&](auto i) { return GenerateLowerDim<Tuple, Number<i>>(tuple); },
                           Number<Tuple::Size()>{});
        const auto upper_dims =
            generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<Tuple::Size()>{});

        return transform_tensor_descriptor(desc, transforms, lower_dims, upper_dims);
    }

    template <typename LayoutShape, typename LayoutStrides>
    static auto MakeDescriptor(const LayoutShape shape, const LayoutStrides strides)
    {
        const auto unrolled_shape   = UnrollNestedTuple(shape);
        const auto unrolled_strides = UnrollNestedTuple(strides);

        if constexpr(ck::is_same_v<LayoutStrides, Tuple<>>)
        {
            const auto desc = make_naive_tensor_descriptor_packed(unrolled_shape);
            return MakeMerges(shape, desc);
        }
        else
        {
            static_assert(unrolled_shape.Size() == unrolled_strides.Size(),
                          "Size of strides and shape are not consistent.");
            const auto desc = make_naive_tensor_descriptor(unrolled_shape, unrolled_strides);
            return MakeMerges(shape, desc);
        }
    }

    public:
    using Descriptor = remove_cvref_t<decltype(MakeDescriptor(Shape{}, Strides{}))>;

    /**
     * \brief Layout constructor.
     *
     * \param shape Shape for layout.
     * \param strides Strides for layout (optional if tensor is packed).
     * \return Layout object.
     */
    __host__ __device__ Layout() = delete;
    __host__ __device__ Layout(const Shape shape, const Strides strides) : descriptor_{}
    {
        if constexpr(!Descriptor::IsKnownAtCompileTime())
        {
            descriptor_ = MakeDescriptor(shape, strides);
        }
    }

    __host__ __device__ Layout(const Shape shape) : descriptor_{}
    {
        if constexpr(!Descriptor::IsKnownAtCompileTime())
        {
            descriptor_ = MakeDescriptor(shape, Strides{});
        }
    }

    // Returns real offset to element
    template <typename Tuple>
    __host__ __device__ constexpr index_t operator()(const Tuple Idx) const
    {
        return descriptor_.CalculateOffset(Idx);
    }

    template <typename Tuple>
    __host__ __device__ constexpr index_t operator()(const Tuple Idx)
    {
        return descriptor_.CalculateOffset(Idx);
    }

    // Upper dim getter
    template <index_t IDim>
    __host__ __device__ constexpr index_t GetLength() const
    {
        return descriptor_.GetLength(Number<IDim>{});
    }

    template <index_t IDim>
    __host__ __device__ constexpr index_t GetLength()
    {
        return descriptor_.GetLength(Number<IDim>{});
    }

    private:
    Descriptor descriptor_;
};

// Upper dim getter
template <index_t idx, typename L>
index_t size(L layout)
{
    return layout.template GetLength<idx>();
}

template <typename Shape, typename Strides>
Layout<Shape, Strides> make_layout(const Shape& shape, const Strides& strides)
{
    return Layout<Shape, Strides>(shape, strides);
}

template <typename Shape>
Layout<Shape> make_layout(const Shape& shape)
{
    return Layout<Shape>(shape);
}

} // namespace tensor_transform_wrapper
} // namespace ck
