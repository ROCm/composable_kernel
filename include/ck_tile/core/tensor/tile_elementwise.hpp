// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/null_tensor.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// TODO: support tensors with different distribution
template <typename InOutElementFunc,
          typename... InOutDstrTensors,
          typename = std::enable_if_t<std::conjunction_v<
              std::negation<std::is_same<std::remove_const_t<InOutDstrTensors>, null_tensor>>...>>>
CK_TILE_DEVICE void tile_elementwise_inout(const InOutElementFunc& inout_element_func,
                                           InOutDstrTensors&... inout_dstr_tensors)
{
    // TODO: make sure all distributed tensors have same lengths and distribution
    // static_assert(xxx);

    constexpr index_t thread_buffer_size =
        __type_pack_element<0, InOutDstrTensors...>::get_thread_buffer_size();

    static_for<0, thread_buffer_size, 1>{}(
        [&](auto i) { inout_element_func(inout_dstr_tensors.get_thread_buffer().at(i)...); });
}

template <typename InElementFunc,
          typename... InDstrTensors,
          typename = std::enable_if_t<
              std::conjunction_v<std::negation<std::is_same<InDstrTensors, null_tensor>>...>>>
CK_TILE_DEVICE auto tile_elementwise_in(const InElementFunc& in_element_func,
                                        const InDstrTensors&... in_dstr_tensors)
{
    using OutDataType = decltype(in_element_func(typename InDstrTensors::DataType{}...));

    // TODO: make sure all distributed tensors have same lengths and distribution
    // static_assert(xxx);
    constexpr auto in_tile_dstr = __type_pack_element<0, InDstrTensors...>::get_tile_distribution();

    constexpr index_t thread_buffer_size =
        __type_pack_element<0, InDstrTensors...>::get_thread_buffer_size();

    auto out_dstr_tensor = make_static_distributed_tensor<OutDataType>(in_tile_dstr);

    static_for<0, thread_buffer_size, 1>{}([&](auto i) {
        out_dstr_tensor.get_thread_buffer()(i) =
            in_element_func(in_dstr_tensors.get_thread_buffer()[i]...);
    });

    return out_dstr_tensor;
}

template <typename DstrTensors, typename T>
CK_TILE_DEVICE void set_tile(DstrTensors& dstr_tensor, const T& value)
{
    tile_elementwise_inout(
        [&value](auto& x) {
            x = type_convert<typename DstrTensors::DataType, remove_cvref_t<T>>(value);
        },
        dstr_tensor);
}

template <typename T>
CK_TILE_DEVICE void set_tile(null_tensor&, const T&)
{
}

// TODO: prefer to use per-dword value to set a tensor, in case compiler not doing well with
// sub-dword tensor...
template <typename DstrTensors, index_t v>
CK_TILE_DEVICE void set_tile(DstrTensors& dstr_tensor, number<v>)
{
    constexpr index_t tensor_bytes =
        DstrTensors::get_thread_buffer_size() * sizeof(typename DstrTensors::DataType);
    if constexpr(v == 0 && tensor_bytes % 4 == 0)
    {
        using dvec_t = array<index_t, tensor_bytes / 4>;
        auto& tensor = reinterpret_cast<dvec_t&>(dstr_tensor.get_thread_buffer());
        for(auto i = 0; i < tensor.size(); i++)
            tensor.get(i) = v;
    }
    else
    {
        tile_elementwise_inout(
            [](auto& x) { x = type_convert<typename DstrTensors::DataType, index_t>(v); },
            dstr_tensor);
    }
}

template <index_t v>
CK_TILE_DEVICE void set_tile(null_tensor&, number<v>)
{
}

template <typename DstrTensors>
CK_TILE_DEVICE void clear_tile(DstrTensors& dstr_tensor)
{
    set_tile(dstr_tensor, 0);
}

// TODO: this is ugly
template <typename OutDataType, typename InDstrTensors>
CK_TILE_DEVICE auto cast_tile_pk_fp8x4(const InDstrTensors& in_dstr_tensors)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // This API is designed to use the _pk_ serious of function
    constexpr auto in_tile_dstr = InDstrTensors::get_tile_distribution();

    constexpr index_t thread_buffer_size = InDstrTensors::get_thread_buffer_size();
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
        uint32_t x = __builtin_amdgcn_cvt_pk_fp8_f32(
            in_dstr_tensors.get_thread_buffer()[number<4 * i + 0>{}],
            in_dstr_tensors.get_thread_buffer()[number<4 * i + 1>{}],
            dummy_old,
            false); // false -> WORD0

        uint32_t y = __builtin_amdgcn_cvt_pk_fp8_f32(
            in_dstr_tensors.get_thread_buffer()[number<4 * i + 2>{}],
            in_dstr_tensors.get_thread_buffer()[number<4 * i + 3>{}],
            dummy_old,
            false); // false -> WORD0

        constexpr int32_t m0 = 0x05040100;
        using vec_t          = array<OutDataType, 4>;

        vec_t d = bit_cast<vec_t>(__builtin_amdgcn_perm(y, x, m0));
        out_dstr_tensor.get_thread_buffer().template set_as<vec_t>(number<i>{}, d);
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
CK_TILE_DEVICE auto cast_tile(const SrcDstrTensors& src_tensor)
{
    if constexpr((std::is_same_v<DstType, fp8_t> ||
                  std::is_same_v<DstType, bf8_t>)&&std::is_same_v<typename SrcDstrTensors::DataType,
                                                                  float> &&
                 (SrcDstrTensors::get_thread_buffer_size() % 4 == 0))
    {
        return cast_tile_pk_fp8x4<DstType, SrcDstrTensors>(src_tensor);
    }
    else
        return tile_elementwise_in(type_convert<DstType, typename SrcDstrTensors::DataType>,
                                   src_tensor);
}

// no-op function for null_tensor arguments
template <typename InOutElementFunc,
          typename... MaybeNullTensor,
          typename = std::enable_if_t<
              std::disjunction_v<std::is_same<remove_cvref_t<MaybeNullTensor>, null_tensor>...>>>
CK_TILE_DEVICE void tile_elementwise_inout(const InOutElementFunc&, MaybeNullTensor&&...)
{
}

// no-op function for null_tensor arguments
template <typename InElementFunc,
          typename... MaybeNullTensor,
          typename = std::enable_if_t<
              std::disjunction_v<std::is_same<remove_cvref_t<MaybeNullTensor>, null_tensor>...>>>
CK_TILE_DEVICE auto tile_elementwise_in(const InElementFunc&, MaybeNullTensor&&...)
{
    return null_tensor{};
}

} // namespace ck_tile
