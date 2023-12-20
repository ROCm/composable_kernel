// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../utils/tensor_utils.hpp"

namespace ck {
namespace wrapper {

/**
 * \brief Perform generic copy between two tensors. Tensors must have the
 *  same size.
 *
 * \param src_tensor Source tensor.
 * \param dst_tensor Destination tensor.
 */
template <typename SrcTensorType, typename DstTensorType>
__host__ __device__ void copy(const SrcTensorType& src_tensor, DstTensorType& dst_tensor)
{
    assert(size(src_tensor) == size(dst_tensor));
    using SrcSizeTensor = decltype(size(src_tensor));
    using DstSizeTensor = decltype(size(dst_tensor));
    if constexpr(is_known_at_compile_time<SrcSizeTensor>::value)
    {
        static_for<0, SrcSizeTensor{}, 1>{}([&](auto i) { dst_tensor(i) = src_tensor(i); });
    }
    else if constexpr(is_known_at_compile_time<DstSizeTensor>::value)
    {
        static_for<0, DstSizeTensor{}, 1>{}([&](auto i) { dst_tensor(i) = src_tensor(i); });
    }
    else
    {
        for(int i = 0; i < size(src_tensor); i++)
        {
            dst_tensor(i) = src_tensor(i);
        }
    }
}

} // namespace wrapper
} // namespace ck
