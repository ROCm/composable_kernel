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
 * \tparam BufferAddressSpace Memory type (Generic, Global, Lds, Vgpr, Sgpr).
 * \tparam ElementType Element data type.
 * \tparam Shape Tensor shape (layout component).
 * \tparam Strides Tensor strides (layout component).
 * \tparam NumVectors Number of vectors (only for Vgpr, Sgpr).
 * \tparam ScalarPerVector Scalars per vector (only for Vgpr, Sgpr).
 */
template <AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides,
          index_t NumVectors,     // param for Register memory
          index_t ScalarPerVector // param for Register memory
          >
struct Tensor
{
    private:
    // Check if Tuple containt Slice object
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
    // It is need to calculate offset for new tensor
    template <typename... Ts>
    constexpr auto GetStartIdxForSlicedTensor(const Tuple<Ts...>& idx)
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
                    // if slice, return begging of the interval
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
    constexpr auto GetShapeFromSlicedTensor(const Tuple<Ts...>& idx, const ShapeTmpType& shape)
    {
        // Pack each value in tuple to remove empty tuples after generation
        auto new_shape = generate_tuple(
            [&](auto i) {
                constexpr auto num_i = Number<i>{};
                if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {
                    if constexpr(!IsSlicing(tuple_element_t<i.value, Tuple<Ts...>>{}))
                    {
                        // if tuple does not have then we can remove dimension
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
                                              const StridesTmpType& strides)
    {
        // Pack each value in tuple to remove empty tuples after generation
        auto new_strides = generate_tuple(
            [&](auto i) {
                constexpr auto num_i = Number<i>{};
                if constexpr(is_detected<is_tuple, tuple_element_t<i.value, Tuple<Ts...>>>::value)
                {
                    if constexpr(!IsSlicing(tuple_element_t<i.value, Tuple<Ts...>>{}))
                    {
                        // if tuple does not have then we can remove dimension
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

    static constexpr AddressSpaceEnum TensorBufferAddressSpace = BufferAddressSpace;
    static constexpr bool IsDynamicBuffer = !(BufferAddressSpace == AddressSpaceEnum::Sgpr ||
                                              BufferAddressSpace == AddressSpaceEnum::Vgpr);

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

    __host__ __device__ constexpr auto& GetLayout() const { return layout_; }

    // Getter for new sliced tensor
    template <typename... Ts, enable_if_t<IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ auto operator[](const Tuple<Ts...>& idx)
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

    // Getter for new sliced tensor, call operator[]
    template <typename... Ts, enable_if_t<IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ auto operator()(const Tuple<Ts...>& idx)
    {
        return this->operator[](idx);
    }

    // Getter for new sliced tensor, call operator[]
    template <typename... Idxs, enable_if_t<IsSlicing(Tuple<Idxs...>{}), bool> = false>
    __host__ __device__ auto operator()(Idxs... idxs)
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

    // Getter for the value reference, call operator[]
    template <typename... Ts, enable_if_t<!IsSlicing(Tuple<Ts...>{}), bool> = false>
    __host__ __device__ ElementType& operator()(const Tuple<Ts...>& idx)
    {
        return this->operator[](idx);
    }

    // Getter for the value reference, call operator[]
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
    // If not regsiter, use dynamic buffer, if no static buffer
    using Buffer = std::conditional_t<IsDynamicBuffer, DynamicBufferType, StaticBufferType>;

    const Layout<Shape, Strides> layout_;
    Buffer buffer_;
};

} // namespace wrapper
} // namespace ck
