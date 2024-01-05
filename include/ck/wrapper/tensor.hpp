// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "utils/tensor_utils.hpp"
#include "utils/tensor_partition.hpp"
#include "utils/layout_utils.hpp"

namespace ck {
namespace wrapper {

/**
 * \brief Tensor wrapper that performs static and dynamic buffer logic.
 *
 * \tparam BufferAddressSpace Memory type (Generic, Global, LDS, VGPR, SGPR).
 * \tparam ElementType Element data type.
 * \tparam Shape Tensor shape (layout component).
 * \tparam UnnestedDescriptorType Unnested descriptor (layout component).
 * \tparam NumVectors Number of vectors (only for VGPR, SGPR).
 * \tparam ScalarPerVector Scalars per vector (only for VGPR, SGPR).
 */
template <MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename UnnestedDescriptorType,
          index_t NumVectors,     // param for Register memory
          index_t ScalarPerVector // param for Register memory
          >
struct Tensor
{
    private:
    // Check if Tuple contains Slice object
    template <typename T>
    __host__ __device__ constexpr static bool IsSlicing(T&&)
    {
        return is_detected<is_slice, T>::value;
    }
    template <typename... Ts>
    __host__ __device__ constexpr static bool IsSlicing(Tuple<Ts...>&&)
    {
        return (IsSlicing(Ts{}) || ...);
    }

    // Calculate new tensor shape after slice
    template <typename... Ts, typename ShapeTmpType>
    __host__ __device__ constexpr auto GetShapeFromSlicedTensor(const Tuple<Ts...>& idx,
                                                                const ShapeTmpType& shape) const
    {
        // Pack each value in tuple to remove empty tuples after generation
        auto new_shape = generate_tuple(
            [&](auto i) {
                constexpr auto num_i = Number<i>{};
                if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {
                    if constexpr(!IsSlicing(tuple_element_t<i.value, Tuple<Ts...>>{}))
                    {
                        // if tuple does not have any slice then we can remove dimension
                        return Tuple<>{};
                    }
                    else
                    {
                        // if tuple then recurrence
                        return make_tuple(GetShapeFromSlicedTensor(idx.At(num_i), shape.At(num_i)));
                    }
                }
                else if constexpr(is_detected<is_slice,
                                              tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {
                    // calculate new dimension
                    const auto& dim = size(shape.At(num_i));
                    const auto val  = idx.At(num_i).range(dim);
                    return make_tuple(val);
                }
                else
                {
                    // remove dimension for just value
                    return Tuple<>{};
                }
            },
            Number<Tuple<Ts...>::Size()>{});
        // Remove empty tuples (deleted elements) and return
        return UnrollNestedTuple<0, 1>(new_shape);
    }

    // Generate Freeze for each of nested shape
    template <typename T, typename ShapeTmpType>
    __host__ __device__ constexpr auto GenerateMultipleFreeze(T idx,
                                                              const ShapeTmpType& shape) const
    {
        const auto unrolled_shape = UnrollNestedTuple(shape);
        return generate_tuple(
            [&](auto i) {
                // dimension offset from idx
                const auto dim     = unrolled_shape.At(Number<i>{});
                const auto dim_idx = idx % dim;
                idx /= dim;
                return make_freeze_transform(dim_idx);
            },
            Number<decltype(unrolled_shape)::Size()>{});
    }

    template <typename... Ts, typename ShapeTmpType>
    __host__ __device__ constexpr auto
    GetTransformsFromSlicedTensor(const Tuple<Ts...>& idx, const ShapeTmpType& shape) const
    {
        // Pack each value in tuple to remove empty tuples after generation
        auto transforms = generate_tuple(
            [&](auto i) {
                constexpr auto num_i = Number<i>{};
                if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {
                    return GetTransformsFromSlicedTensor(idx.At(num_i), shape.At(num_i));
                }
                else if constexpr(is_detected<is_slice,
                                              tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {

                    const auto from  = idx.At(num_i).from_;
                    const auto dim   = shape.At(num_i);
                    const auto range = idx.At(num_i).range(dim);
                    return make_slice_transform(range, from, from + range);
                }
                else
                {
                    // remove dimension for just value
                    return GenerateMultipleFreeze(idx.At(num_i), shape.At(num_i));
                }
            },
            Number<Tuple<Ts...>::Size()>{});
        // Remove empty tuples (deleted elements) and return
        return UnrollNestedTuple(transforms);
    }

    // There is no output for Freeze transform
    template <index_t i, typename LowerIndex>
    __host__ __device__ constexpr auto GetSequenceVal(const ck::Freeze<LowerIndex>&) const
    {
        return Sequence<>{};
    }

    template <index_t i, typename LowLength, typename SliceBegin, typename SliceEnd>
    __host__ __device__ constexpr auto
    GetSequenceVal(const ck::Slice<LowLength, SliceBegin, SliceEnd>&) const
    {
        return Sequence<i>{};
    }

    template <index_t i>
    __host__ __device__ constexpr auto GenerateUpperDims(const Tuple<>&) const
    {
        return Tuple<>{};
    }

    template <index_t i, typename... Transforms>
    __host__ __device__ constexpr auto
    GenerateUpperDims(const Tuple<Transforms...>& transforms) const
    {
        constexpr auto num_transforms = Tuple<Transforms...>::Size();
        // Deduce Sequence element for specific transform
        const auto currect_elem = GetSequenceVal<i>(transforms.At(Number<0>{}));
        if constexpr(is_same_v<decltype(currect_elem), const Sequence<>>)
        {
            const auto next_tuple = GenerateUpperDims<i>(TupleSlice<1, num_transforms>(transforms));
            return concat_tuple(make_tuple(currect_elem), next_tuple);
        }
        else
        {
            // Increase i if current_elem is Slice transform
            const auto next_tuple =
                GenerateUpperDims<i + 1>(TupleSlice<1, num_transforms>(transforms));
            return concat_tuple(make_tuple(currect_elem), next_tuple);
        }
    }

    template <typename... Ts, typename ShapeTmpType, typename FlattenDescriptor>
    __host__ __device__ constexpr auto
    GetDescriptorFromSlicedTensor(const Tuple<Ts...>& idx,
                                  const ShapeTmpType& shape,
                                  const FlattenDescriptor& flatten_desc) const
    {
        constexpr auto old_shape_dims = decltype(UnrollNestedTuple(shape))::Size();

        const auto transforms     = GetTransformsFromSlicedTensor(idx, shape);
        using TransformsTupleType = decltype(transforms);

        const auto lower_dims =
            generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<old_shape_dims>{});
        const auto upper_dims = decltype(GenerateUpperDims<0>(TransformsTupleType{})){};
        return transform_tensor_descriptor(flatten_desc, transforms, lower_dims, upper_dims);
    }

    public:
    using ElementSpaceSize  = decltype(Layout<Shape, UnnestedDescriptorType>{
        Shape{}, UnnestedDescriptorType{}}.GetElementSpaceSize()); // SpaceSize type for buffer
    using TensorElementType = ElementType;                          // DataType

    static constexpr MemoryTypeEnum TensorBufferAddressSpace = BufferAddressSpace;
    static constexpr bool IsDynamicBuffer = !(BufferAddressSpace == MemoryTypeEnum ::Sgpr ||
                                              BufferAddressSpace == MemoryTypeEnum ::Vgpr);

    __host__ __device__ Tensor() = delete;
    __host__ __device__ Tensor(ElementType* pointer,
                               const Layout<Shape, UnnestedDescriptorType>& layout)
        : layout_(layout),
          buffer_(make_dynamic_buffer<BufferAddressSpace>(pointer, layout.GetElementSpaceSize()))
    {
    }

    __host__ __device__ Tensor(const Layout<Shape, UnnestedDescriptorType>& layout)
        : layout_(layout)
    {
        static_assert(!IsDynamicBuffer, "Wrong BufferAddressSpace for register.");
    }

    __host__ __device__ constexpr const Layout<Shape, UnnestedDescriptorType>& GetLayout() const
    {
        return layout_;
    }

    // Getter for new sliced tensor
    template <typename... Ts, enable_if_t<IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ auto operator[](const Tuple<Ts...>& idx) const
    {
        static_assert(IsDynamicBuffer, "Register slice is not supported");
        const auto& shape = layout_.GetShape();
        auto new_shape    = GetShapeFromSlicedTensor(idx, shape);

        const auto& flatten_desc = layout_.GetUnnestedDescriptor();
        auto new_desc            = GetDescriptorFromSlicedTensor(idx, shape, flatten_desc);
        const auto new_layout =
            Layout<decltype(new_shape), decltype(new_desc)>(new_shape, new_desc);
        return make_tensor<BufferAddressSpace>(buffer_.p_data_, new_layout);
    }

    template <typename... Ts, enable_if_t<IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ auto operator()(const Tuple<Ts...>& idx) const
    {
        return this->operator[](idx);
    }

    template <typename... Idxs, enable_if_t<IsSlicing(Tuple<Idxs...>{}), bool> = false>
    __host__ __device__ auto operator()(Idxs... idxs) const
    {
        return this->operator[](make_tuple(idxs...));
    }

    // Getter for the const value
    template <typename... Ts, enable_if_t<!IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ const ElementType& operator[](const Tuple<Ts...>& idx) const
    {
        if constexpr(IsDynamicBuffer)
        {
            const index_t offset = layout_(idx);
            return buffer_[offset];
        }
        else
        {
            constexpr index_t offset = Layout<Shape, UnnestedDescriptorType>{
                Shape{},
                UnnestedDescriptorType{}}.template operator()<Tuple<Ts...>>();
            return buffer_[Number<offset>{}];
        }
    }

    template <typename... Ts, enable_if_t<!IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ const ElementType& operator()(const Tuple<Ts...>& idx) const
    {
        return this->operator[](idx);
    }

    template <typename... Idxs, enable_if_t<!IsSlicing(Tuple<Idxs...>{}), bool> = false>
    __host__ __device__ const ElementType& operator()(Idxs... idxs) const
    {
        return this->operator[](make_tuple(idxs...));
    }

    // Getter for the value reference
    template <typename... Ts, enable_if_t<!IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ ElementType& operator[](const Tuple<Ts...>& idx)
    {
        if constexpr(IsDynamicBuffer)
        {
            const index_t offset = layout_(idx);
            return buffer_(offset);
        }
        else
        {
            constexpr index_t offset = Layout<Shape, UnnestedDescriptorType>{
                Shape{},
                UnnestedDescriptorType{}}.template operator()<Tuple<Ts...>>();
            return buffer_(Number<offset>{});
        }
    }

    template <typename... Ts, enable_if_t<!IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ ElementType& operator()(const Tuple<Ts...>& idx)
    {
        return this->operator[](idx);
    }

    template <typename... Idxs, enable_if_t<!IsSlicing(Tuple<Idxs...>{}), bool> = false>
    __host__ __device__ ElementType& operator()(Idxs... idxs)
    {
        return this->operator[](make_tuple(idxs...));
    }

    __host__ __device__ constexpr auto GetDefaultDescriptor()
    {
        return layout_.GetDefaultDescriptor();
    }

    __host__ __device__ ElementType* GetPointer() const { return buffer_.p_data_; }

    private:
    using DynamicBufferType = DynamicBuffer<BufferAddressSpace,
                                            ElementType,
                                            ElementSpaceSize,
                                            true /*InvalidElementUseNumericalZeroValue*/>;
    using StaticBufferType =
        StaticBufferTupleOfVector<BufferAddressSpace,
                                  ElementType,
                                  NumVectors,
                                  ScalarPerVector,
                                  true /*InvalidElementUseNumericalZeroValue*/>;
    // If register use static buffer, else use dynamic buffer
    using Buffer = std::conditional_t<IsDynamicBuffer, DynamicBufferType, StaticBufferType>;

    const Layout<Shape, UnnestedDescriptorType> layout_;
    Buffer buffer_;
};

} // namespace wrapper
} // namespace ck
