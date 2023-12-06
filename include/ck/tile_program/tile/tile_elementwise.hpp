// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

// TODO: support tensors with different distribution
template <typename InOutElementFunc, typename... InOutDstrTensors>
__device__ void tile_elementwise_inout(const InOutElementFunc& inout_element_func,
                                       InOutDstrTensors&... inout_dstr_tensors)
{
    // TODO: make sure all distributed tensors have same lengths and distribution
    // static_assert(xxx);

    constexpr index_t thread_buffer_size =
        type_pack_element<0, InOutDstrTensors...>::GetThreadBufferSize();

    static_for<0, thread_buffer_size, 1>{}(
        [&](auto i) { inout_element_func(inout_dstr_tensors.GetThreadBuffer().At(i)...); });
}

template <typename InElementFunc, typename... InDstrTensors>
__device__ auto tile_elementwise_in(const InElementFunc& in_element_func,
                                    const InDstrTensors&... in_dstr_tensors)
{
    using OutDataType = decltype(in_element_func(typename InDstrTensors::DataType{}...));

    // TODO: make sure all distributed tensors have same lengths and distribution
    // static_assert(xxx);
    constexpr auto in_tile_dstr = type_pack_element<0, InDstrTensors...>::GetTileDistribution();

    constexpr index_t thread_buffer_size =
        type_pack_element<0, InDstrTensors...>::GetThreadBufferSize();

    auto out_dstr_tensor = make_static_distributed_tensor<OutDataType>(in_tile_dstr);

    static_for<0, thread_buffer_size, 1>{}([&](auto i) {
        out_dstr_tensor.GetThreadBuffer()(i) =
            in_element_func(in_dstr_tensors.GetThreadBuffer()[i]...);
    });

    return out_dstr_tensor;
}

template <typename DstrTensors, typename T>
__device__ void set_tile(DstrTensors& dstr_tensor, const T& value)
{
    tile_elementwise_inout(
        [&value](auto& x) {
            x = type_convert<typename DstrTensors::DataType, remove_cvref_t<T>>(value);
        },
        dstr_tensor);
}

template <typename DstrTensors>
__device__ void clear_tile(DstrTensors& dstr_tensor)
{
    set_tile(dstr_tensor, 0);
}

template <typename DstType, typename SrcDstrTensors>
__device__ auto cast_tile(const SrcDstrTensors& src_tensor)
{
    return tile_elementwise_in(type_convert<DstType, typename SrcDstrTensors::DataType>,
                               src_tensor);
}

} // namespace tile_program
} // namespace ck
