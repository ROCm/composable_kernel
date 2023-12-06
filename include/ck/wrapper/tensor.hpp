// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "layout.hpp"
#include "utils/tensor_utils.hpp"

namespace ck {
namespace wrapper {

template <AddressSpaceEnum BufferAddressSpace,
          typename ElementType,
          typename Shape,
          typename Strides>
struct Tensor
{
    using ElementSpaceSize = decltype(Layout<Shape, Strides>{
        Shape{}, Strides{}}.GetElementSpaceSize());

    __host__ __device__ Tensor() = delete;
    __host__ __device__ Tensor(ElementType* pointer, const Layout<Shape, Strides>& layout)
        : layout_(layout),
          dynamic_buffer_(
              make_dynamic_buffer<BufferAddressSpace>(pointer, layout.GetElementSpaceSize()))
    {
    }

    template <typename... Ts>
    __host__ __device__ index_t operator[](const Tuple<Ts...>& Idx) const
    {
        // Padding is not supported, so we can assume that read should be valid.
        return dynamic_buffer_.template Get<ElementType>(layout_(Idx), true /*is_valid*/);
    }

    template <typename... Ts>
    __host__ __device__ index_t operator()(const Tuple<Ts...>& Idx) const
    {
        return dynamic_buffer_.template Get<ElementType>(layout_(Idx), true /*is_valid*/);
    }

    template <typename... Idxs>
    __host__ __device__ index_t operator()(Idxs... idxs) const
    {
        const auto idxs_tuple = make_tuple(idxs...);
        return dynamic_buffer_.template Get<ElementType>(layout_(idxs_tuple), true /*is_valid*/);
    }

    private:
    const Layout<Shape, Strides>& layout_;
    DynamicBuffer<BufferAddressSpace,
                  ElementType,
                  ElementSpaceSize,
                  true /*InvalidElementUseNumericalZeroValue*/>
        dynamic_buffer_;
};

} // namespace wrapper
} // namespace ck
