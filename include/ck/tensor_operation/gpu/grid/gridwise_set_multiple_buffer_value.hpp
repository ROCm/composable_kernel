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
#ifndef CK_GRIDWISE_SET_MULTIPLE_BUFFER_VALUE_HPP
#define CK_GRIDWISE_SET_MULTIPLE_BUFFER_VALUE_HPP

#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename Grid1dBufferDescType,
          index_t NumBuffer,
          index_t BlockSize,
          typename DataTypePointerTuple,
          typename DataTypeTuple>
__global__ void kernel_multiple_buffer_set_value(const Grid1dBufferDescType grid_1d_buffer_desc,
                                                 DataTypePointerTuple p_global_tuple,
                                                 DataTypeTuple value_tuple)

{
    static_assert(NumBuffer == DataTypePointerTuple::Size() && NumBuffer == DataTypeTuple::Size(),
                  "The tuple size should be same as NumBuffer!");

    static_for<0, NumBuffer, 1>{}([&](auto iB) {
        using DataTypePointer     = remove_cvref_t<decltype(DataTypePointerTuple{}[iB])>;
        using DataTypeFromPointer = remove_pointer_t<DataTypePointer>;
        using DataType            = remove_cvref_t<decltype(DataTypeTuple{}[iB])>;

        static_assert(is_same<DataType, DataTypeFromPointer>::value,
                      "Types in tuples does not match!");
    });

    constexpr auto I0 = Number<0>{};

    const index_t thread_local_id = get_thread_local_1d_id();
    const index_t block_global_id = get_block_1d_id();

    const index_t thread_global_id = block_global_id * BlockSize + thread_local_id;

    auto value_buf_tuple = generate_tuple(
        [&](auto iB) {
            using DataType = remove_cvref_t<decltype(DataTypeTuple{}[iB])>;

            return StaticBuffer<AddressSpaceEnum::Vgpr, DataType, 1, true>{};
        },
        Number<NumBuffer>{});

    static_for<0, NumBuffer, 1>{}([&](auto iB) {
        static_for<0, 1, 1>{}([&](auto J) { value_buf_tuple(iB)(J) = value_tuple[iB]; });
    });

    auto global_buf_tuple = generate_tuple(
        [&](auto iB) {
            return make_dynamic_buffer<AddressSpaceEnum::Global>(
                p_global_tuple(iB), grid_1d_buffer_desc.GetElementSpaceSize());
        },
        Number<NumBuffer>{});

    constexpr auto val_buff_desc = make_naive_tensor_descriptor_packed(make_tuple(Number<1>{}));

    static_for<0, NumBuffer, 1>{}([&](auto iB) {
        using DataType      = remove_cvref_t<decltype(DataTypeTuple{}[iB])>;
        using PassThroughOp = tensor_operation::element_wise::PassThrough;

        auto threadwise_store = ThreadwiseTensorSliceTransfer_v1r3<DataType,
                                                                   DataType,
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
                             value_buf_tuple(iB),
                             grid_1d_buffer_desc,
                             global_buf_tuple(iB));
    });
};

} // namespace ck
#endif
