// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/type_convert.hpp"

#define TEST_FLAG 0

using ck::e8m0_bexp_t;
using ck::float2_t;

using ck::f4_t;
using ck::f4x2_pk_t;
using ck::f4x2_t;

constexpr uint64_t test_size = 2;

inline __device__ f4x2_t convert(float2_t x, float scale)
{
    constexpr int seed = 1254739;
    uint32_t rng = ck::prand_generator<float, seed>(reinterpret_cast<size_t>(&x), x[0]);
    union
    {
        uint32_t bitwise;
        f4x2_t f4x2_array[4];
    } value{0};
// apply a temporary workaround for gfx950
#if TEST_FLAG
    uint8_t l     = ck::utils::sat_convert_to_type_sr<f4_t>(x[1] / scale, rng);
    uint8_t h     = ck::utils::sat_convert_to_type_sr<f4_t>(x[0] / scale, rng);
    value.bitwise = (h << 4) | l;
#else
    // permute high bits and low bits to match the order of the original vector
    value.bitwise = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(
        value.bitwise, float2_t{x[1], x[0]}, rng, scale, 0);
#endif // TEST_FLAG
    return value.f4x2_array[0];
}

__device__ void
test_convert(uint64_t N, float* p_test)
{
    uint64_t i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    // f32x2 -> f4x2
    float2_t f32x2 = {1.0f, -4.0f};
    auto scale2 = e8m0_bexp_t(2.0f);

    f4x2_t f4x2 = convert(f32x2, type_convert<float>(scale2)); // expect {0.5, -2}

    p_test[i++] = ck::type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<0>{}))); // 0.5f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = ck::type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<1>{}))); // -2.0f
    if(i >= N)
    {
        return;
    }
}

__global__ void test_convert_device(uint64_t N, float* p_test)
{
    test_convert(N, p_test);
}

TEST(MXFP4, DeviceScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);

    DeviceMem device_out(test_size * sizeof(float));

    test_convert_device<<<1, 1>>>(
        test_size,
        static_cast<float*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    auto i = 0;

    // SR
    EXPECT_EQ(out[i++], 0.5f);
    EXPECT_EQ(out[i++], -2.0f);
}
