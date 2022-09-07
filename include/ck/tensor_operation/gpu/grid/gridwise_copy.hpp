// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseCopyFunctor,
          typename InGrid1dDesc,
          typename OutGrid1dDesc,
          typename InDataTypePointer,
          typename OutDataTypePointer,
          typename ElementwiseOperation>
__global__ void kernel_nd_copy(const InGrid1dDesc in_grid_1d_desc,
                               const OutGrid1dDesc out_grid_1d_desc,
                               const InDataTypePointer p_in_global,
                               const OutDataTypePointer p_out_global,
                               const ElementwiseOperation elementwise_op)
{
    GridwiseCopyFunctor::Run(
        in_grid_1d_desc, out_grid_1d_desc, p_in_global, p_out_global, elementwise_op);
}

template <typename InGrid1dDesc,
          typename OutGrid1dDesc,
          typename InDataTypePointer,
          typename OutDataTypePointer,
          typename ElementwiseOperation,
          index_t MPerThread,
          index_t InScalarPerVector,
          index_t OutScalarPerVector>
struct GridwiseCopy
{
    static constexpr auto I0 = Number<0>{};

    static constexpr auto thread_buffer_desc_m =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MPerThread>{}));

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    __device__ static void Run(const InGrid1dDesc in_grid_1d_desc,
                               const OutGrid1dDesc out_grid_1d_desc,
                               const InDataTypePointer p_in_global,
                               const OutDataTypePointer p_out_global,
                               const ElementwiseOperation elementwise_op)
    {
        const index_t thread_global_id = get_thread_global_1d_id();

        using InDataType   = remove_cv_t<remove_pointer_t<InDataTypePointer>>;
        auto in_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr, InDataType, MPerThread, true>{};

        using OutDataType   = remove_cv_t<remove_pointer_t<OutDataTypePointer>>;
        auto out_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr, OutDataType, MPerThread, true>{};

        auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_global, in_grid_1d_desc.GetElementSpaceSize());

        auto out_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_global, out_grid_1d_desc.GetElementSpaceSize());

        const auto thread_global_offset = make_multi_index(thread_global_id * MPerThread);

        const index_t blockSize    = get_block_size();
        const index_t blockPerGrid = get_grid_size();
        const auto M               = in_grid_1d_desc.GetLength(I0);
        const index_t loop_step    = blockPerGrid * blockSize * MPerThread;
        const auto loop_step_index = make_multi_index(loop_step);

        auto in_global_load =
            ThreadwiseTensorSliceTransfer_v2<InDataType,
                                             InDataType,
                                             decltype(in_grid_1d_desc),
                                             decltype(thread_buffer_desc_m),
                                             Sequence<MPerThread>, // SliceLengths
                                             Sequence<0>,          // DimAccessOrder
                                             0,                    // SrcVectorDim
                                             InScalarPerVector,    // ScalarPerVector
                                             1,                    // SrcScalarStrideInVector
                                             false>{in_grid_1d_desc, thread_global_offset};

        auto out_global_store =
            ThreadwiseTensorSliceTransfer_v1r3<OutDataType,
                                               OutDataType,
                                               decltype(thread_buffer_desc_m),
                                               decltype(out_grid_1d_desc),
                                               PassThroughOp,
                                               Sequence<MPerThread>, // SliceLengths
                                               Sequence<0>,          // DimAccessOrder
                                               0,                    // SrcVectorDim
                                               OutScalarPerVector,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               false>(
                out_grid_1d_desc, thread_global_offset, PassThroughOp{});

        index_t num_iter = M / (loop_step);
        do
        {
            in_global_load.Run(in_grid_1d_desc,
                               in_global_buf,
                               thread_buffer_desc_m,
                               make_tuple(I0),
                               in_thread_buf);

            in_global_load.MoveSrcSliceWindow(in_grid_1d_desc, loop_step_index);

            static_for<0, MPerThread, 1>{}([&](auto iM) {
                // get reference to in data
                const auto& in_data_ref = in_thread_buf(iM);

                // get reference to dst data
                auto& out_data_ref = out_thread_buf(iM);

                elementwise_op(out_data_ref, in_data_ref);
            });

            out_global_store.Run(thread_buffer_desc_m,
                                 make_tuple(I0),
                                 out_thread_buf,
                                 out_grid_1d_desc,
                                 out_global_buf);

            out_global_store.MoveDstSliceWindow(out_grid_1d_desc, loop_step_index);
        } while(--num_iter);
    }
};

} // namespace ck
