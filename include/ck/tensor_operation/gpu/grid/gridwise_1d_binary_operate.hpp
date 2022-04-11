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
#ifndef CK_GRIDWISE_1D_BINARY_OPERATE_HPP
#define CK_GRIDWISE_1D_BINARY_OPERATE_HPP

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize,
          typename InDataType_X0,
          typename InDataType_X1,
          typename OutDataType,
          typename AccDataType,
          typename Grid1dBufferDescType,
          typename BinaryOperatorType>
__global__ void kernel_1d_binary_operate(const Grid1dBufferDescType grid_1d_buffer_desc,
                                         BinaryOperatorType binary_op,
                                         const InDataType_X0* const __restrict__ p_in_x0,
                                         const InDataType_X1* const __restrict__ p_in_x1,
                                         OutDataType* const __restrict__ p_out_y)
{
    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    constexpr auto I0 = Number<0>{};

    const index_t thread_local_id = get_thread_local_1d_id();
    const index_t block_global_id = get_block_1d_id();

    const index_t thread_global_id = block_global_id * BlockSize + thread_local_id;

    StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, 1, true> value_buf_x0;
    StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, 1, true> value_buf_x1;
    StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, 1, true> value_buf_y;

    constexpr auto val_buff_desc = make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

    auto global_buf_x0 = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_in_x0, grid_1d_buffer_desc.GetElementSpaceSize());
    auto global_buf_x1 = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_in_x1, grid_1d_buffer_desc.GetElementSpaceSize());
    auto global_buf_y = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_out_y, grid_1d_buffer_desc.GetElementSpaceSize());

    if(thread_global_id < grid_1d_buffer_desc.GetElementSize())
    {
        auto threadwise_load_x0 = ThreadwiseTensorSliceTransfer_v2<InDataType_X0,
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

        auto threadwise_load_x1 = ThreadwiseTensorSliceTransfer_v2<InDataType_X1,
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

        threadwise_load_x0.Run(
            grid_1d_buffer_desc, global_buf_x0, val_buff_desc, make_tuple(I0), value_buf_x0);
        threadwise_load_x1.Run(
            grid_1d_buffer_desc, global_buf_x1, val_buff_desc, make_tuple(I0), value_buf_x1);

        binary_op(value_buf_y(I0), value_buf_x0[I0], value_buf_x1[I0]);

        auto threadwise_store = ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                                   OutDataType,
                                                                   decltype(val_buff_desc),
                                                                   Grid1dBufferDescType,
                                                                   PassThroughOp,
                                                                   Sequence<1>,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperationEnum_t::Set,
                                                                   1,
                                                                   true>(
            grid_1d_buffer_desc, make_multi_index(thread_global_id), PassThroughOp{});

        threadwise_store.Run(
            val_buff_desc, make_tuple(I0), value_buf_y, grid_1d_buffer_desc, global_buf_y);
    }
};

} // namespace ck
#endif
