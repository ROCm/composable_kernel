// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::e8m0_bexp_t;
using ck::float16_t;
using ck::float2_t;
using ck::float32_t;
using ck::scaled_type_convert;
using ck::type_convert;

using ck::f4_convert_rne;
using ck::f4_convert_sr;
using ck::f4_t;
using ck::f4x16_t;
using ck::f4x2_pk_t;
using ck::f4x2_t;
using ck::f4x32_t;

__host__ __device__ void test_mx_fp4_to_fp32(float* p_test)
{
    /// Test vector conversions
    // f4x2 -> f32x2
    f4x2_t f4x2{f4x2_t::data_v{0b00011100}}; // 0b0001(=0.5) and 0b1100(=-2.0)
    auto scale2 = e8m0_bexp_t(2.0f);

    float2_t f32x2 = scaled_type_convert<float2_t>(scale2, f4x2);
    p_test[0]      = f32x2[0];
    p_test[1]      = f32x2[1];
}

__global__ void run_test_mx_fp4_to_fp32(float* p_test) { test_mx_fp4_to_fp32(p_test); }

__host__ __device__ void test_mx_fp32_to_fp4_rne(float* p_test)
{
    // f32x2 -> f4x2
    float2_t f32x2 = {1.0f, -4.0f};
    auto scale2    = e8m0_bexp_t(2.0f);
    f4x2_t f4x2    = f4_convert_rne(f32x2, type_convert<float>(scale2)); // expect {0.5, -2}

    p_test[0] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<0>{}))); // 0.5f
    p_test[1] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<1>{}))); // -2.0f
}

__global__ void run_test_mx_fp32_to_fp4_rne(float* p_test) { test_mx_fp32_to_fp4_rne(p_test); }

__host__ __device__ void test_mx_fp32_to_fp4_sr(float* p_test)
{
    float2_t f32x2 = {1.0f, -4.0f};
    auto scale2    = e8m0_bexp_t(2.0f);
    f4x2_t f4x2    = f4_convert_sr(f32x2, type_convert<float>(scale2)); // expect {0.5, -2}

    p_test[0] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<0>{}))); // 0.5f
    p_test[1] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<1>{}))); // -2.0f
}

__global__ void run_test_mx_fp32_to_fp4_sr(float* p_test) { test_mx_fp32_to_fp4_sr(p_test); }

__host__ __device__ void test_mx_fp32_to_fp4_sr_failing(float* p_test)
{
    float2_t f32x2 = {1.0f, -4.0f};
    auto scale2    = e8m0_bexp_t(2.0f);
    f4x2_t f4x2 = ck::f4_convert_sr_repro(f32x2, type_convert<float>(scale2)); // expect {0.5, -2}

    p_test[0] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<0>{}))); // 0.5f
    p_test[1] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<1>{}))); // -2.0f
}

__global__ void run_test_mx_fp32_to_fp4_sr_failing(float* p_test)
{
    test_mx_fp32_to_fp4_sr_failing(p_test);
}

TEST(MXFP4, FP4ToFP32)
{
    std::vector<float> out(2, -1.0f);

    DeviceMem device_out(2 * sizeof(float));

    run_test_mx_fp4_to_fp32<<<1, 1>>>(static_cast<float*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    // f4x2 -> f32x2
    EXPECT_EQ(out[0], 1.0f);
    EXPECT_EQ(out[1], -4.0f);
}

TEST(MXFP4, FP32ToFP4RNE)
{
    std::vector<float> out(2, -1.0f);

    DeviceMem device_out(2 * sizeof(float));

    run_test_mx_fp32_to_fp4_rne<<<1, 1>>>(static_cast<float*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    // f32x2 -> f4x2
    // RNE
    EXPECT_EQ(out[0], 0.5f);
    EXPECT_EQ(out[1], -2.0f);
}

TEST(MXFP4, FP32ToFP4SR)
{
    std::vector<float> out(2, -1.0f);

    DeviceMem device_out(2 * sizeof(float));

    run_test_mx_fp32_to_fp4_sr<<<1, 1>>>(static_cast<float*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    // SR
    EXPECT_EQ(out[0], 0.5f);
    EXPECT_EQ(out[1], -2.0f);
}

TEST(MXFP4, FP32ToFP4SRFailing)
{
    std::vector<float> out(2, -1.0f);

    DeviceMem device_out(2 * sizeof(float));

    run_test_mx_fp32_to_fp4_sr_failing<<<1, 1>>>(static_cast<float*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    // SR
    EXPECT_EQ(out[0], 0.5f);
    EXPECT_EQ(out[1], -2.0f);
}
