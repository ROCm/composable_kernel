/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_GRIDWISE_1D_COMPUTE_INV_VARIANCE_RUNNING_MEAN_AND_VARIANCE_FUSED_HPP
#define CK_GRIDWISE_1D_COMPUTE_INV_VARIANCE_RUNNING_MEAN_AND_VARIANCE_FUSED_HPP

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize,
          typename AccDataType,
          typename Grid1dBufferDescType,
          typename BinaryOperationInvVariance,
          typename BinaryOperationMovingAverage>
__global__ void kernel_1d_compute_inv_variance_running_mean_and_variance(
    const Grid1dBufferDescType grid_1d_buffer_desc,
    BinaryOperationInvVariance op_invVariance,
    BinaryOperationMovingAverage op_movingAverage,
    const AccDataType* const __restrict__ p_mean,
    const AccDataType* const __restrict__ p_meansquare,
    AccDataType* const __restrict__ p_invVariance,
    AccDataType* const __restrict__ p_runningMean,
    AccDataType* const __restrict__ p_runningVariance)
{
    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    constexpr auto I0 = Number<0>{};

    const index_t thread_local_id = get_thread_local_1d_id();
    const index_t block_global_id = get_block_1d_id();

    const index_t thread_global_id = block_global_id * BlockSize + thread_local_id;

    StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, 1, true> mean_buf;
    StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, 1, true> meansquare_buf;
    StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, 1, true> invVariance_buf;
    StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, 1, true> runningMean_buf;
    StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, 1, true> runningVariance_buf;

    constexpr auto val_buff_desc = make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

    auto global_mean_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        p_mean, grid_1d_buffer_desc.GetElementSpaceSize());
    auto global_meansquare_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        p_meansquare, grid_1d_buffer_desc.GetElementSpaceSize());
    auto global_invVariance_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        p_invVariance, grid_1d_buffer_desc.GetElementSpaceSize());
    auto global_runningMean_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        p_runningMean, grid_1d_buffer_desc.GetElementSpaceSize());
    auto global_runningVariance_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
        p_runningVariance, grid_1d_buffer_desc.GetElementSpaceSize());

    if(thread_global_id < grid_1d_buffer_desc.GetElementSize())
    {
        auto threadwise_load = ThreadwiseTensorSliceTransfer_v2<AccDataType,
                                                                AccDataType,
                                                                Grid1dBufferDescType,
                                                                decltype(val_buff_desc),
                                                                Sequence<1>,
                                                                Sequence<0>,
                                                                0,
                                                                1,
                                                                1,
                                                                false>(
            grid_1d_buffer_desc, make_multi_index(thread_global_id));

        threadwise_load.Run(
            grid_1d_buffer_desc, global_mean_buf, val_buff_desc, make_tuple(I0), mean_buf);
        threadwise_load.Run(grid_1d_buffer_desc,
                            global_meansquare_buf,
                            val_buff_desc,
                            make_tuple(I0),
                            meansquare_buf);
        threadwise_load.Run(grid_1d_buffer_desc,
                            global_runningMean_buf,
                            val_buff_desc,
                            make_tuple(I0),
                            runningMean_buf);
        threadwise_load.Run(grid_1d_buffer_desc,
                            global_runningVariance_buf,
                            val_buff_desc,
                            make_tuple(I0),
                            runningVariance_buf);

        AccDataType variance;

        variance = meansquare_buf[I0] - mean_buf[I0] * mean_buf[I0];

        op_invVariance(invVariance_buf(I0), mean_buf(I0), meansquare_buf(I0));
        op_movingAverage(runningMean_buf(I0), runningMean_buf(I0), mean_buf(I0));
        op_movingAverage(runningVariance_buf(I0), runningVariance_buf(I0), variance);

        auto threadwise_store = ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                                   AccDataType,
                                                                   decltype(val_buff_desc),
                                                                   Grid1dBufferDescType,
                                                                   PassThroughOp,
                                                                   Sequence<1>,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperationEnum::Set,
                                                                   1,
                                                                   true>(
            grid_1d_buffer_desc, make_multi_index(thread_global_id), PassThroughOp{});

        threadwise_store.Run(val_buff_desc,
                             make_tuple(I0),
                             invVariance_buf,
                             grid_1d_buffer_desc,
                             global_invVariance_buf);
        threadwise_store.Run(val_buff_desc,
                             make_tuple(I0),
                             runningMean_buf,
                             grid_1d_buffer_desc,
                             global_runningMean_buf);
        threadwise_store.Run(val_buff_desc,
                             make_tuple(I0),
                             runningVariance_buf,
                             grid_1d_buffer_desc,
                             global_runningVariance_buf);
    }
};

} // namespace ck
#endif
