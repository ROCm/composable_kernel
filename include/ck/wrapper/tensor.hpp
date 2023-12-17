// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "utils/tensor_utils.hpp"
#include "utils/layout_utils.hpp"

namespace ck {
namespace wrapper {

/**
 * \brief Tensor wrapper that performs static and dynamic buffer logic.
 *
 * \tparam BufferAddressSpace Memory type (Generic, Global, LDS, VGPR, SGPR).
 * \tparam ElementType Element data type.
 * \tparam Shape Tensor shape (layout component).
 * \tparam Strides Tensor strides (layout component).
 * \tparam NumVectors Number of vectors (only for VGPR, SGPR).
 * \tparam ScalarPerVector Scalars per vector (only for VGPR, SGPR).
 */
template <MemoryTypeEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,     // param for Register memory
          index_t ScalarPerVector // param for Register memory
          >
struct Tensor
{
    private:
    // Check if Tuple contains Slice object
    template <typename T>
    constexpr static bool IsSlicing(T&&)
    {
        return is_detected<is_slice, T>::value;
    }
    template <typename... Ts>
    constexpr static bool IsSlicing(Tuple<Ts...>&&)
    {
        return (IsSlicing(Ts{}) || ...);
    }

    // Calculate first index of new tensor after slice
    // It is needed to calculate offset for new tensor
    template <typename... Ts>
    constexpr auto GetStartIdxForSlicedTensor(const Tuple<Ts...>& idx) const
    {
        const auto start_idx_for_sliced_tensor = generate_tuple(
            [&](auto i) {
                constexpr auto num_i = Number<i>{};
                if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {
                    // if tuple then recurrence
                    return GetStartIdxForSlicedTensor(idx.At(num_i));
                }
                else if constexpr(is_detected<is_slice,
                                              tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {
                    // if slice, return the beginning of the interval
                    return idx.At(num_i).from_;
                }
                else
                {
                    // if one dim selected
                    return idx.At(num_i);
                }
            },
            Number<Tuple<Ts...>::Size()>{});

        return start_idx_for_sliced_tensor;
    }

    // Calculate new tensor shape after slice
    template <typename... Ts, typename ShapeTmpType>
    constexpr auto GetShapeFromSlicedTensor(const Tuple<Ts...>& idx,
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

    template <typename... Ts, typename StridesTmpType>
    constexpr auto GetStridesFromSlicedTensor(const Tuple<Ts...>& idx,
                                              const StridesTmpType& strides) const
    {
        // Pack each value in tuple to remove empty tuples after generation
        auto new_strides = generate_tuple(
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
                        return make_tuple(
                            GetStridesFromSlicedTensor(idx.At(num_i), strides.At(num_i)));
                    }
                }
                else if constexpr(is_detected<is_slice,
                                              tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {
                    // Stride will be the same
                    return make_tuple(strides.At(num_i));
                }
                else
                {
                    // remove dimension for just value
                    return Tuple<>{};
                }
            },
            Number<Tuple<Ts...>::Size()>{});
        // Remove empty tuples (deleted elements) and return
        return UnrollNestedTuple<0, 1>(new_strides);
    }

    public:
    using ElementSpaceSize  = decltype(Layout<Shape, Strides>{
        Shape{}, Strides{}}.GetElementSpaceSize()); // SpaceSize type for buffer
    using TensorElementType = ElementType;           // DataType

    static constexpr MemoryTypeEnum TensorBufferAddressSpace = BufferAddressSpace;
    static constexpr bool IsDynamicBuffer = !(BufferAddressSpace == MemoryTypeEnum ::Sgpr ||
                                              BufferAddressSpace == MemoryTypeEnum ::Vgpr);

    __host__ __device__ Tensor() = delete;
    __host__ __device__ Tensor(ElementType* pointer, const Layout<Shape, Strides>& layout)
        : layout_(layout),
          buffer_(make_dynamic_buffer<BufferAddressSpace>(pointer, layout.GetElementSpaceSize()))
    {
    }

    __host__ __device__ Tensor(const Layout<Shape, Strides>& layout) : layout_(layout)
    {
        static_assert(!IsDynamicBuffer, "Wrong BufferAddressSpace for register.");
    }

    __host__ __device__ constexpr const Layout<Shape, Strides>& GetLayout() const
    {
        return layout_;
    }

    // Getter for new sliced tensor
    template <typename... Ts, enable_if_t<IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ auto operator[](const Tuple<Ts...>& idx) const
    {
        static_assert(IsDynamicBuffer, "Register slice is not supported");
        // Calculate offset based on first idx for new tensor
        const index_t offset = layout_(GetStartIdxForSlicedTensor(idx));

        auto new_shape = GetShapeFromSlicedTensor(idx, layout_.GetShape());
        if constexpr(is_same_v<Strides, Tuple<>>)
        {
            auto new_layout = make_layout(new_shape);
            return make_tensor<BufferAddressSpace>(buffer_.p_data_ + offset, new_layout);
        }
        else
        {
            auto new_strides = GetStridesFromSlicedTensor(idx, layout_.GetStrides());
            auto new_layout  = make_layout(new_shape, new_strides);
            return make_tensor<BufferAddressSpace>(buffer_.p_data_ + offset, new_layout);
        }
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
            if constexpr(is_same_v<Strides, Tuple<>>)
            {
                constexpr index_t offset =
                    Layout<Shape, Strides>{Shape{}}.template operator()<Tuple<Ts...>>();
                return buffer_[Number<offset>{}];
            }
            else
            {
                constexpr index_t offset =
                    Layout<Shape, Strides>{Shape{}, Strides{}}.template operator()<Tuple<Ts...>>();
                return buffer_[Number<offset>{}];
            }
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
            if constexpr(is_same_v<Strides, Tuple<>>)
            {
                constexpr index_t offset =
                    Layout<Shape, Strides>{Shape{}}.template operator()<Tuple<Ts...>>();
                return buffer_(Number<offset>{});
            }
            else
            {
                constexpr index_t offset =
                    Layout<Shape, Strides>{Shape{}, Strides{}}.template operator()<Tuple<Ts...>>();
                return buffer_(Number<offset>{});
            }
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

    const Layout<Shape, Strides> layout_;
    Buffer buffer_;
};

} // namespace wrapper
} // namespace ck
