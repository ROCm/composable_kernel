// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/null_tensor.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

// TODO: support tensors with different distribution
template <typename InOutElementFunc,
          typename... InOutDstrTensors,
          typename = std::enable_if_t<std::conjunction_v<
              std::negation<std::is_same<std::remove_const_t<InOutDstrTensors>, NullTensor>>...>>>
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

template <typename InElementFunc,
          typename... InDstrTensors,
          typename = std::enable_if_t<
              std::conjunction_v<std::negation<std::is_same<InDstrTensors, NullTensor>>...>>>
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

template <typename T>
__device__ void set_tile(NullTensor&, const T&)
{
}

template <typename DstrTensors>
__device__ void clear_tile(DstrTensors& dstr_tensor)
{
    set_tile(dstr_tensor, 0);
}

template <typename OutDataType, typename InDstrTensors>
__device__ auto cast_tile_pk_fp8x4(const InDstrTensors& in_dstr_tensors)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // This API is designed to use the _pk_ serious of function
    constexpr auto in_tile_dstr = InDstrTensors::GetTileDistribution();

    constexpr index_t thread_buffer_size = InDstrTensors::GetThreadBufferSize();
    static_assert(thread_buffer_size % 4 == 0);
    constexpr index_t thread_buffer_size_pk = thread_buffer_size / 4;

    auto out_dstr_tensor = make_static_distributed_tensor<OutDataType>(in_tile_dstr);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
    // __builtin_amdgcn_cvt_pk_fp8_f32() this builtin require the old value, and
    // will generate a v_mov_b32 vxxx [old] before cvt, which result in unwanted ISA
    // so we prepare an uninitialized variable purposely, and turn off the warning
    int dummy_old;
    static_for<0, thread_buffer_size_pk, 1>{}([&](auto i) {
        uint32_t x =
            __builtin_amdgcn_cvt_pk_fp8_f32(in_dstr_tensors.GetThreadBuffer()[Number<4 * i + 0>{}],
                                            in_dstr_tensors.GetThreadBuffer()[Number<4 * i + 1>{}],
                                            dummy_old,
                                            false); // false -> WORD0

        uint32_t y =
            __builtin_amdgcn_cvt_pk_fp8_f32(in_dstr_tensors.GetThreadBuffer()[Number<4 * i + 2>{}],
                                            in_dstr_tensors.GetThreadBuffer()[Number<4 * i + 3>{}],
                                            dummy_old,
                                            false); // false -> WORD0

        constexpr int32_t m0 = 0x05040100;
        using vec_t          = typename vector_type<OutDataType, 4>::type;

        vec_t d = bit_cast<vec_t>(__builtin_amdgcn_perm(y, x, m0));
        out_dstr_tensor.GetThreadBuffer().template SetAsType<vec_t>(Number<4 * i>{}, d);
    });
#pragma clang diagnostic pop

    return out_dstr_tensor;
#else
    // fallback
    return tile_elementwise_in(type_convert<OutDataType, typename InDstrTensors::DataType>,
                               in_dstr_tensors);
#endif
}

template <typename DstType, typename SrcDstrTensors>
__device__ auto cast_tile(const SrcDstrTensors& src_tensor)
{
    if constexpr((ck::is_same_v<DstType, f8_t> ||
                  ck::is_same_v<DstType, bf8_t>)&&ck::is_same_v<typename SrcDstrTensors::DataType,
                                                                float> &&
                 (SrcDstrTensors::GetThreadBufferSize() % 4 == 0))
    {
        return cast_tile_pk_fp8x4<DstType, SrcDstrTensors>(src_tensor);
    }
    else
        return tile_elementwise_in(type_convert<DstType, typename SrcDstrTensors::DataType>,
                                   src_tensor);
}

// no-op function for NullTensor arguments
template <typename InOutElementFunc,
          typename... MaybeNullTensor,
          typename = std::enable_if_t<
              std::disjunction_v<std::is_same<remove_cvref_t<MaybeNullTensor>, NullTensor>...>>>
__device__ void tile_elementwise_inout(const InOutElementFunc&, MaybeNullTensor&&...)
{
}

// no-op function for NullTensor arguments
template <typename InElementFunc,
          typename... MaybeNullTensor,
          typename = std::enable_if_t<
              std::disjunction_v<std::is_same<remove_cvref_t<MaybeNullTensor>, NullTensor>...>>>
__device__ auto tile_elementwise_in(const InElementFunc&, MaybeNullTensor&&...)
{
    return NullTensor{};
}

} // namespace tile_program
} // namespace ck
