
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <ck_tile/host/hip_check_error.hpp>

#define HSTU_CHECK(COND, ERR)                  \
    if(!(COND))                                \
    {                                          \
        std::ostringstream ostr;               \
        ostr << "'" #COND "' failed: " << ERR; \
        throw std::runtime_error(ostr.str());  \
    }

static inline int get_number_of_cu()
{
    int device;

    HIP_CHECK_ERROR(hipGetDevice(&device));

    hipDeviceProp_t props;

    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));

    return props.multiProcessorCount;
}

namespace ck_tile {

namespace detail {

// A helper struct for detecting kUseTrLoad
// T is the pipeline class used by the kernel instance
template <typename T, typename = void>
struct has_use_trload_flag : std::false_type
{
};

template <typename T>
struct has_use_trload_flag<
    T,
    std::enable_if_t<std::is_convertible_v<decltype(T::kUseTrLoad), bool> && T::kUseTrLoad>>
    : std::true_type
{
};

template <typename T>
static inline constexpr bool is_using_trload_v = has_use_trload_flag<T>::value;

// scale is uniform (scalar register), c is per-lane (vector register)
// GFX9 VOP2: V_MUL_F32 VDST, SRC0, SRC1 - SRC0 can be SGPR, SRC1 must be VGPR

// scale is uniform (scalar register), c is per-lane (vector register)
// GFX9 VOP2: V_MUL_F32 VDST, SRC0, SRC1 - SRC0 can be SGPR, SRC1 must be VGPR
CK_TILE_DEVICE static void v_mul_f32_two(float& c0, float& c1, float scale)
{
    asm volatile("v_mul_f32 %[v_c0], %[s_scale], %[v_c0] \n\
                  v_mul_f32 %[v_c1], %[s_scale], %[v_c1]"
                 : [v_c0] "+v"(c0), [v_c1] "+v"(c1)
                 : [s_scale] "s"(scale)
                 :);
}

CK_TILE_DEVICE static void v_mul_f32(float& c, float scale)
{
    asm volatile("v_mul_f32 %[v_c], %[s_scale], %[v_c]" : [v_c] "+v"(c) : [s_scale] "s"(scale) :);
}

CK_TILE_DEVICE fp32x2_t pk_mul_f32(fp32x2_t lhs, fp32x2_t rhs)
{
    fp32x2_t result;
    asm volatile("v_pk_mul_f32 %[result], %[lhs], %[rhs]"
                 : [result] "=v"(result)
                 : [lhs] "v"(lhs), [rhs] "v"(rhs));
    return result;
}

template <typename InOutDstrTensor>
CK_TILE_DEVICE static void scale_tile_in_scalar(InOutDstrTensor& in_out_dstr_tensor, float scale)
{
    using DataType = typename InOutDstrTensor::DataType;

    if constexpr(std::is_same_v<std::remove_cv_t<DataType>, float>)
    {
        auto tmp_scale = type_convert<DataType>(scale);

        constexpr index_t thread_buffer_size = InOutDstrTensor::get_thread_buffer_size();

        static_for<0, thread_buffer_size, 2>{}([&](auto idx) {
            v_mul_f32_two(in_out_dstr_tensor.thread_buf_[idx],
                          in_out_dstr_tensor.thread_buf_[idx + 1],
                          tmp_scale);
        });
    }
    else
    {
        tile_elementwise_inout([&scale](auto& x) { x = x * scale; }, in_out_dstr_tensor);
    };
};

template <typename InOutDstrTensor>
CK_TILE_DEVICE static void scale_tile_in_pack(InOutDstrTensor& in_out_dstr_tensor, float scale)
{
    using DataType = typename InOutDstrTensor::DataType;

    if constexpr(std::is_same_v<std::remove_cv_t<DataType>, float>)
    {
        fp32x2_t pk_scale;

        pk_scale.x = scale;
        pk_scale.y = scale;

        constexpr index_t thread_buffer_size = InOutDstrTensor::get_thread_buffer_size();

        static_for<0, thread_buffer_size, 2>{}([&](auto idx) {
            fp32x2_t input                          = {in_out_dstr_tensor.thread_buf_[idx],
                              in_out_dstr_tensor.thread_buf_[idx + 1]};
            auto output                             = pk_mul_f32(input, pk_scale);
            in_out_dstr_tensor.thread_buf_[idx]     = output.x;
            in_out_dstr_tensor.thread_buf_[idx + 1] = output.y;
        });
    }
    else
    {
        tile_elementwise_inout([&scale](auto& x) { x = x * scale; }, in_out_dstr_tensor);
    };
};

} // namespace detail

} // namespace ck_tile
