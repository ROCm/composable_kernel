// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../utils/tensor_utils.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace wrapper {

/**
 * \brief Perform generic copy between two tensors partitions (threadwise copy).
 *  Tensors must have the same size.
 *
 * \param src_tensor Source tensor.
 * \param dst_tensor Destination tensor.
 */
template <typename SrcTensorType, typename DstTensorType>
__host__ __device__ void copy(const SrcTensorType& src_tensor, DstTensorType& dst_tensor)
{
    if constexpr(!SrcTensorType::IsDynamicBuffer)
    {
        using SizeType = decltype(size(src_tensor));
        static_for<0, SizeType{}, 1>{}([&](auto i) { dst_tensor(i) = src_tensor(i); });
    }
    else if constexpr(!DstTensorType::IsDynamicBuffer)
    {
        using SizeType = decltype(size(dst_tensor));
        static_for<0, SizeType{}, 1>{}([&](auto i) { dst_tensor(i) = src_tensor(i); });
    }
    else
    {
        for(int i = 0; i < size(src_tensor); i++)
        {
            dst_tensor(i) = src_tensor(i);
        }
    }
}

/**
 * \brief Perform optimized copy between two tensors partitions (threadwise copy).
 * Tensors must have the same size.
 *
 * \tparam DimAccessOrderTuple Tuple with dimension access order.
 * \tparam VectorDim Dimension for vectorized read and write.
 * \tparam ScalarPerVector Number of scalar per vectorized read and write.
 * \param src_tensor Source tensor.
 * \param dst_tensor Destination tensor.
 */
template <typename DimAccessOrderTuple,
          index_t VectorDim,
          index_t ScalarPerVector,
          typename SrcTensorType,
          typename DstTensorType>
__device__ void copy(const SrcTensorType& src_tensor, DstTensorType& dst_tensor)
{
    static_assert(is_detected<is_tuple, DimAccessOrderTuple>::value);
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    const auto& in_grid_desc  = layout(src_tensor).GetUnrolledDescriptor();
    const auto& out_grid_desc = layout(dst_tensor).GetUnrolledDescriptor();

    using SrcShapeType         = remove_cvref_t<decltype(shape(src_tensor))>;
    constexpr index_t num_dims = SrcShapeType::Size();

    constexpr auto thread_slice_lengths =
        generate_sequence_v2([](auto I) { return size(SrcShapeType{}.At(I)); }, Number<num_dims>{});
    constexpr auto dim_access_order = generate_sequence_v2(
        [](auto I) { return DimAccessOrderTuple{}.At(I); }, Number<num_dims>{});

    if constexpr(SrcTensorType::IsDynamicBuffer && DstTensorType::IsDynamicBuffer)
    {
        // Perform a copy between DynamicBuffers
        auto transfer = ThreadwiseTensorSliceTransfer_v7<
            Tuple<typename SrcTensorType::TensorElementType>,
            Tuple<typename DstTensorType::TensorElementType>,
            decltype(tie(in_grid_desc)),
            decltype(tie(out_grid_desc)),
            tensor_operation::element_wise::PassThrough,
            Sequence<static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
            decltype(thread_slice_lengths),
            decltype(dim_access_order),
            VectorDim,
            ScalarPerVector,
            Sequence<false>,
            Sequence<false>>{in_grid_desc,
                             make_tuple(src_tensor.GetMultiIdxOffsets()),
                             out_grid_desc,
                             make_tuple(dst_tensor.GetMultiIdxOffsets()),
                             tensor_operation::element_wise::PassThrough{}};

        transfer.Run(tie(in_grid_desc),
                     tie(src_tensor.GetBuffer()),
                     tie(out_grid_desc),
                     tie(dst_tensor.GetBuffer()));
    }
    else if constexpr(!SrcTensorType::IsDynamicBuffer && DstTensorType::IsDynamicBuffer)
    {
        // Perform copy from StaticBuffer to DynamicBuffer
        const auto src_slice_origin_idxs =
            generate_tuple([&](auto) { return I0; }, Number<num_dims>{});

        auto transfer =
            ThreadwiseTensorSliceTransfer_v1r3<typename SrcTensorType::TensorElementType,
                                               typename DstTensorType::TensorElementType,
                                               remove_cvref_t<decltype(in_grid_desc)>,
                                               remove_cvref_t<decltype(out_grid_desc)>,
                                               tensor_operation::element_wise::PassThrough,
                                               decltype(thread_slice_lengths),
                                               decltype(dim_access_order),
                                               VectorDim,
                                               ScalarPerVector,
                                               InMemoryDataOperationEnum::Set,
                                               I1,
                                               true>{out_grid_desc,
                                                     dst_tensor.GetMultiIdxOffsets(),
                                                     tensor_operation::element_wise::PassThrough{}};

        transfer.Run(in_grid_desc,
                     src_slice_origin_idxs,
                     src_tensor.GetBuffer(),
                     out_grid_desc,
                     dst_tensor.GetBuffer());
    }
    else if constexpr(SrcTensorType::IsDynamicBuffer && !DstTensorType::IsDynamicBuffer)
    {
        // Perform copy from DynamicBuffer to StaticBuffer
        const auto src_dst_slice_origin =
            generate_tuple([&](auto) { return I0; }, Number<num_dims>{});
        constexpr auto src_vector_tensor_lengths = generate_sequence_v2(
            [&](auto I) {
                if constexpr(I == VectorDim)
                {
                    return Number<ScalarPerVector>{};
                }
                else
                {
                    return I1;
                }
            },
            Number<num_dims>{});

        auto transfer =
            ThreadwiseTensorSliceTransfer_v4r1<typename SrcTensorType::TensorElementType,
                                               typename DstTensorType::TensorElementType,
                                               remove_cvref_t<decltype(in_grid_desc)>,
                                               remove_cvref_t<decltype(out_grid_desc)>,
                                               decltype(thread_slice_lengths),
                                               decltype(dim_access_order),
                                               decltype(src_vector_tensor_lengths),
                                               decltype(dim_access_order)>{
                src_tensor.GetMultiIdxOffsets()};

        transfer.Run(in_grid_desc,
                     src_dst_slice_origin,
                     src_tensor.GetBuffer(),
                     out_grid_desc,
                     src_dst_slice_origin,
                     dst_tensor.GetBuffer());
    }
    else
    {
        // Perform copy between StaticBuffers
        copy(src_tensor, dst_tensor);
    }
}

} // namespace wrapper
} // namespace ck
