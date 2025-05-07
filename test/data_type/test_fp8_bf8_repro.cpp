// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/type_convert.hpp"

#define TEST_FLAG 0

using ck::half_t;
using ck::half2_t;

using ck::f8_ocp_t;
using ck::f8x2_t;
using ck::bf8_ocp_t;
using ck::bf8x2_t;
using ck::fp8_storage_t;

using ck::type_convert;

typedef fp8_storage_t fp8x2_storage_t __attribute__((ext_vector_type(2)));
typedef short shortx2_t __attribute__((ext_vector_type(2)));

constexpr uint64_t test_size = 2;

inline __device__ f8x2_t convert_fp8(half2_t x)
{
#if TEST_FLAG
    return fp8x2_storage_t{
        ck::fp8_impl::cvt_half_t_to_fp8<ck::ck_fp8_interpretation_t::CK_E4M3_OCP>(x[0]),
        ck::fp8_impl::cvt_half_t_to_fp8<ck::ck_fp8_interpretation_t::CK_E4M3_OCP>(x[1])};
#else
    union
    {
        half2_t half_vec;
        shortx2_t i16_vec;
        fp8_storage_t i8val[4];
    } val;

    constexpr shortx2_t i16x2val = {0, 0};
    val.half_vec                 = x;

    val.i16_vec = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(i16x2val, val.half_vec, /* scale */ 1.f, 0);

    return fp8x2_storage_t{val.i8val[0], val.i8val[1]};
#endif
}

__device__ void
test_convert_fp8(uint64_t N, half_t* p_test)
{
    uint64_t i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    half2_t f16x2 = {type_convert<half_t>(1.0f), type_convert<half_t>(-4.0f)};

    f8x2_t f8x2 = convert_fp8(f16x2);

    p_test[i++] = ck::type_convert<half_t>(f8x2.AsType<f8_ocp_t>()(ck::Number<0>{}));
    if(i >= N)
    {
        return;
    }
    p_test[i++] = ck::type_convert<half_t>(f8x2.AsType<f8_ocp_t>()(ck::Number<1>{}));
    if(i >= N)
    {
        return;
    }
}

__global__ void test_convert_fp8_device(uint64_t N, half_t* p_test)
{
    test_convert_fp8(N, p_test);
}

TEST(FP8, DeviceScaledConvert)
{
    std::vector<half_t> out(test_size, type_convert<half_t>(-1.0f));

    DeviceMem device_out(test_size * sizeof(half_t));

    test_convert_fp8_device<<<1, 1>>>(
        test_size,
        static_cast<half_t*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    auto i = 0;

    EXPECT_EQ(out[i++], type_convert<half_t>(1.f));
    EXPECT_EQ(out[i++], type_convert<half_t>(-4.f));
}

inline __device__ bf8x2_t convert_bf8(half2_t x)
{
#if TEST_FLAG
    return fp8x2_storage_t{
        ck::fp8_impl::cvt_half_t_to_fp8<ck::ck_fp8_interpretation_t::CK_E5M2_OCP>(x[0]),
        ck::fp8_impl::cvt_half_t_to_fp8<ck::ck_fp8_interpretation_t::CK_E5M2_OCP>(x[1])};
#else
    union
    {
        half2_t half_vec;
        shortx2_t i16_vec;
        fp8_storage_t i8val[4];
    } val;

    constexpr shortx2_t i16x2val = {0, 0};
    val.half_vec                 = x;

    val.i16_vec = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(i16x2val, val.half_vec, /* scale */ 1.f, 0);

    return fp8x2_storage_t{val.i8val[0], val.i8val[1]};
#endif
}

__device__ void
test_convert_bf8(uint64_t N, half_t* p_test)
{
    uint64_t i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    half2_t f16x2 = {type_convert<half_t>(2.0f), type_convert<half_t>(-8.0f)};

    bf8x2_t bf8x2 = convert_bf8(f16x2);

    p_test[i++] = ck::type_convert<half_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{}));
    if(i >= N)
    {
        return;
    }
    p_test[i++] = ck::type_convert<half_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{}));
    if(i >= N)
    {
        return;
    }
}

__global__ void test_convert_bf8_device(uint64_t N, half_t* p_test)
{
    test_convert_bf8(N, p_test);
}

TEST(BF8, DeviceScaledConvert)
{
    std::vector<half_t> out(test_size, type_convert<half_t>(-1.0f));

    DeviceMem device_out(test_size * sizeof(half_t));

    test_convert_bf8_device<<<1, 1>>>(
        test_size,
        static_cast<half_t*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    auto i = 0;

    EXPECT_EQ(out[i++], type_convert<half_t>(2.f));
    EXPECT_EQ(out[i++], type_convert<half_t>(-8.f));
}
