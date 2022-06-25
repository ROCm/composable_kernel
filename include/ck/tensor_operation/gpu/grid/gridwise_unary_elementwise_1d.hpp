// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseUEltwise,
          typename ADataType,
          typename BDataType,
          typename GridDesc_M0,
          typename ElementwiseFunctor>
__global__ void kernel_unary_elementwise_1d(const ADataType* __restrict__ p_a_global,
                                            BDataType* __restrict__ p_b_global,
                                            const GridDesc_M0 a_grid_desc_m0,
                                            const GridDesc_M0 b_grid_desc_m0,
                                            const ElementwiseFunctor functor)
{
    GridwiseUEltwise::Run(p_a_global, p_b_global, a_grid_desc_m0, b_grid_desc_m0, functor);
}

template <typename ADataType,
          typename BDataType,
          typename GridDesc_M0,
          typename ElementwiseFunctor,
          index_t ScalarPerVector>
struct GridwiseUnaryElementwise_1D
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto thread_desc_m0 =
        make_naive_tensor_descriptor_packed(make_tuple(Number<ScalarPerVector>{}));

    using PassThrough = tensor_operation::element_wise::PassThrough;

    static __device__ auto CalculateElementwiseIndex()
    {
        const index_t global_thread_id = get_thread_global_1d_id();
        return make_multi_index(global_thread_id * ScalarPerVector);
    }

    __host__ __device__ static constexpr bool CheckValidity(const GridDesc_M0 a_grid_desc_m0,
                                                            const GridDesc_M0 b_grid_desc_m0)
    {
        return a_grid_desc_m0.GetLength(I0) == b_grid_desc_m0.GetLength(I0);
    }

    __host__ __device__ static constexpr index_t CalculateGridSize(const index_t tensor_size)
    {
        const index_t grid_size = math::integer_divide_ceil(tensor_size, 256 * ScalarPerVector);

        return grid_size;
    }

    __device__ static void Run(const ADataType* __restrict__ p_a_global,
                               BDataType* __restrict__ p_b_global,
                               const GridDesc_M0 a_grid_desc_m0,
                               const GridDesc_M0 b_grid_desc_m0,
                               const ElementwiseFunctor functor)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_global, a_grid_desc_m0.GetElementSpaceSize());
        auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_global, b_grid_desc_m0.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, ADataType, ScalarPerVector, true> a_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, BDataType, ScalarPerVector, true> b_thread_buf;

        const auto thread_store_global_offset = CalculateElementwiseIndex();

        auto a_global_load =
            ThreadwiseTensorSliceTransfer_v2<ADataType,
                                             ADataType,
                                             GridDesc_M0,
                                             decltype(thread_desc_m0),
                                             Sequence<ScalarPerVector>, // SliceLengths
                                             Sequence<0>,               // DimAccessOrder
                                             0,                         // SrcVectorDim
                                             ScalarPerVector,
                                             1, // SrcScalarStrideInVector
                                             false>{a_grid_desc_m0, thread_store_global_offset};

        auto b_global_write =
            ThreadwiseTensorSliceTransfer_v1r3<BDataType,
                                               BDataType,
                                               decltype(thread_desc_m0),
                                               GridDesc_M0,
                                               PassThrough,
                                               Sequence<ScalarPerVector>, // SliceLengths
                                               Sequence<0>,               // DimAccessOrder
                                               0,                         // DstVectorDim
                                               ScalarPerVector,
                                               InMemoryDataOperationEnum::Set,
                                               1, // DstScalarStrideInVector
                                               false>{
                b_grid_desc_m0, thread_store_global_offset, PassThrough{}};

        const index_t blockSize    = get_block_size();
        const index_t blockPerGrid = get_grid_size();
        const auto m0              = b_grid_desc_m0.GetLength(I0);
        const index_t loop_step    = blockPerGrid * blockSize * ScalarPerVector;
        const auto loop_step_index = make_multi_index(loop_step);

        index_t num_iter = m0 / (loop_step);
        do
        {
            // read and process ScalarPerVector elements
            a_global_load.Run(
                a_grid_desc_m0, a_global_buf, thread_desc_m0, make_tuple(I0), a_thread_buf);

            static_for<0, ScalarPerVector, 1>{}([&](auto m) {
                constexpr auto offset = thread_desc_m0.CalculateOffset(make_tuple(m));
                functor(b_thread_buf(Number<offset>{}), a_thread_buf(Number<offset>{}));
            });

            b_global_write.Run(thread_desc_m0,
                               make_tuple(I0), // SrcSliceOriginIdx
                               b_thread_buf,
                               b_grid_desc_m0,
                               b_global_buf);

            a_global_load.MoveSrcSliceWindow(a_grid_desc_m0, loop_step_index);
            b_global_write.MoveDstSliceWindow(b_grid_desc_m0, loop_step_index);
        } while(--num_iter);
    }
};

} // namespace ck
