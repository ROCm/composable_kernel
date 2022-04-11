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
#ifndef CK_GRIDWISE_1D_UNARY_OPERATE_HPP
#define CK_GRIDWISE_1D_UNARY_OPERATE_HPP

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize,
          typename InOutDataType,
          typename AccDataType,
          typename Grid1dBufferDescType,
          typename UnaryOperatorType>
__global__ void kernel_1d_unary_operate(const Grid1dBufferDescType grid_1d_buffer_desc,
                                        UnaryOperatorType unary_op,
                                        InOutDataType* const __restrict__ p_global)
{
    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    constexpr auto I0 = Number<0>{};

    const index_t thread_local_id = get_thread_local_1d_id();
    const index_t block_global_id = get_block_1d_id();

    const index_t thread_global_id = block_global_id * BlockSize + thread_local_id;

    StaticBuffer<AddressSpaceEnum_t::Vgpr, AccDataType, 1, true> value_buf;

    constexpr auto val_buff_desc = make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

    auto global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_global, grid_1d_buffer_desc.GetElementSpaceSize());

    if(thread_global_id < grid_1d_buffer_desc.GetElementSize())
    {
        auto threadwise_load = ThreadwiseTensorSliceTransfer_v2<InOutDataType,
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
            grid_1d_buffer_desc, global_buf, val_buff_desc, make_tuple(I0), value_buf);

        unary_op(value_buf(I0), value_buf[I0]);

        auto threadwise_store = ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                                   InOutDataType,
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
            val_buff_desc, make_tuple(I0), value_buf, grid_1d_buffer_desc, global_buf);
    }
};

} // namespace ck
#endif
