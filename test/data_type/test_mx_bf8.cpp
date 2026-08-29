// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ::ck::DeviceMem;

using ck::bf8_ocp_t;
using ck::bf8x16_ocp_t;
using ck::bf8x2_ocp_t;
using ck::bf8x32_ocp_t;
using ck::bf8x8_ocp_t;
using ck::bhalf8_t;
using ck::bhalf_t;
using ck::e8m0_bexp_t;
using ck::float16_t;
using ck::float2_t;
using ck::float32_t;
using ck::float8_t;
using ck::half8_t;
using ck::half_t;
using ck::mxf8_convert_rne;
using ck::mxf8_convert_sr;
using ck::scaled_type_convert;
using ck::type_convert;

constexpr uint64_t test_size = 256 * 256 + 2 + 4 + 6 + 16;

/**
 * @brief  Tests conversion of BF8 values to float using E8M0 exponent scaling.
 *
 * This function performs a series of conversions from BF8 values to float values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 and BF8 values,
 * as well as specific vector and rounding conversions.
 *
 * @param N The maximum number of conversions to perform.
 * @param p_test Pointer to the output array where the converted float values will be stored.
 * @param p_completed Pointer to a variable that tracks the number of completed conversions.
 *
 * @note If either p_test or p_completed is nullptr, the function will return immediately.
 * @note The function will stop converting if the number of conversions reaches N.
 * @note First 256*256 conversions are for all possible combinations of E8M0 and BF8 values that are
 * stored in memory sequentially with BF8 values varying faster.
 *
 * The function performs the following conversions:
 * - All possible combinations of E8M0 and BF8 values. [256x256]
 * - Vector conversions bf8x2 -> f32x2. [2]
 * - Vector conversions  f32x2 -> bf8x2 rne. [2]
 * - Vector conversions  f32x2 -> bf8x2 sr. [2]
 * - Round to nearest even conversions for specific float values. [6]
 *
 * The results are stored in the p_test array, and the number of completed conversions
 * is updated in the p_completed variable.
 */
__host__ __device__ void
test_mx_bf8_scaled_convert(uint64_t N, float* p_test, uint64_t* p_completed)
{
    if(p_completed == nullptr)
    {
        return;
    }

    uint64_t& i = *p_completed;
    i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    // All possible combinations of E8M0 and BF8
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
        {
            uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
            auto v          = scaled_type_convert<float>(e8m0_bexp_t(exp_id), bf8_ocp_t{bf8_uid});
            p_test[i]       = v;
            i++;
            if(i >= N)
            {
                return;
            }
        }
    }

    /// Test vector conversions
    // bf8x2 -> f32x2
    bf8x2_ocp_t bf8x2{bf8x2_ocp_t::data_v{0b10000100, 0b00000001}}; //-2^-14, 2^-16
    auto scale = e8m0_bexp_t(8.0f);

    float2_t f32x2 = scaled_type_convert<float2_t>(scale, bf8x2);
    p_test[i++]    = f32x2[0];
    if(i >= N)
    {
        return;
    }
    p_test[i++] = f32x2[1];
    if(i >= N)
    {
        return;
    }

    // f32x2 -> bf8x2
    f32x2       = {-8.0f, 4.0f};
    auto scale2 = e8m0_bexp_t(2.0f);

    bf8x2 = mxf8_convert_rne<bf8x2_ocp_t>(f32x2, type_convert<float>(scale2)); // expect {-4, 2}

    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-4f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 2f
    if(i >= N)
    {
        return;
    }

    auto scale4 = e8m0_bexp_t(4.0f);

    bf8x2 = mxf8_convert_sr<bf8x2_ocp_t>(f32x2, type_convert<float>(scale4)); // expect {-2, 1}

    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-2f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 1f
    if(i >= N)
    {
        return;
    }

    /// Test round to nearest even

    p_test[i++] = type_convert<float>(mxf8_convert_rne<bf8_ocp_t>(1024.0f, 4.0f)); // 1024/4
    if(i >= N)
    {
        return;
    }

    p_test[i++] = type_convert<float>(
        mxf8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::quiet_NaN(), 4.0f)); // => NaN
    if(i >= N)
    {
        return;
    }

    p_test[i++] = type_convert<float>(mxf8_convert_rne<bf8_ocp_t>(
        std::numeric_limits<float>::infinity(), 2.0f)); // => BF8 Inf on device
    if(i >= N)
    {
        return;
    }

    // 31000/0.5 > 57344 => BF8 Inf on device
    p_test[i++] = type_convert<float>(mxf8_convert_rne<bf8_ocp_t>(31000.0f, 0.5f));
    if(i >= N)
    {
        return;
    }

    // -31000/0.5 < -57344  => -BF8 Inf on device
    p_test[i++] = type_convert<float>(mxf8_convert_rne<bf8_ocp_t>(-31000.0f, 0.5f));
    if(i >= N)
    {
        return;
    }

    p_test[i++] = type_convert<float>(
        mxf8_convert_rne<bf8_ocp_t>(powf(2.0f, 16.0f), 4.0f)); // 2^16/4 = 65536/4
    if(i >= N)
    {
        return;
    }

    // test pk8 vector conversion, first /4.0f, then *2.0f
    float fscale = 4.0f;
    float8_t v8_float{2.0f,
                      -4.0f,
                      65536.0f,
                      -65536.0f,
                      std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      powf(2.0f, -16.0f),
                      0.04f};
    // expected {1., -2., 32768., -32768, nan, nan, 0., 0.01953125}
    auto v8_float_back =
        scaled_type_convert<float8_t>(scale2, mxf8_convert_rne<bf8x8_ocp_t>(v8_float, fscale));

    for(int ii = 0; ii < 8; ii++)
    {
        p_test[i++] = v8_float_back[ii];
    }

    // expected {1., -2., 32768., -32768, nan, nan,  0./2^-15, 0.01953125/0.0234375}
    v8_float_back =
        scaled_type_convert<float8_t>(scale2, mxf8_convert_sr<bf8x8_ocp_t>(v8_float, fscale));
    for(int ii = 0; ii < 8; ii++)
    {
        p_test[i++] = v8_float_back[ii];
    }
}

TEST(MXBF8, HostScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);
    uint64_t completed = 0;

    test_mx_bf8_scaled_convert(test_size, out.data(), &completed);

    // V = X * P; X - E8M0 scale, P - BF8

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        auto idx = e8m0_nan_id * 256 + bf8_id;
        ASSERT_TRUE(std::isnan(out[idx]));
    }

    // If P in {Inf, NaN}, then V = P
    std::set<uint8_t> bf8_spec_ids;
    bf8_spec_ids.insert(0b11111111); // -NaN
    bf8_spec_ids.insert(0b01111111); // +NaN
    bf8_spec_ids.insert(0b11111101); // -NaN
    bf8_spec_ids.insert(0b01111101); // +NaN
    bf8_spec_ids.insert(0b11111110); // -NaN
    bf8_spec_ids.insert(0b01111110); // +NaN
    bf8_spec_ids.insert(0b11111100); // -inf
    bf8_spec_ids.insert(0b01111100); // +inf
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(auto bf8_spec_id : bf8_spec_ids)
        {
            auto idx = exp_id * 256 + bf8_spec_id;

            if(std::isnan(type_convert<float>(bf8_ocp_t{bf8_spec_id})))
            {
                ASSERT_TRUE(std::isnan(out[idx]))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id}) << " != " << out[idx];
            }
            else
            {
                ASSERT_EQ(out[idx], type_convert<float>(bf8_ocp_t{bf8_spec_id}))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id}) << " != " << out[idx];
            }
        }
    }

    // V = X * P; X, P - finite
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
        {
            if(bf8_spec_ids.find(bf8_id) != bf8_spec_ids.end())
                continue;

            uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
            auto idx        = exp_id * 256 + bf8_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<float>(e8m0_bexp_t(exp_id)) *
                                type_convert<float>(bf8_ocp_t{bf8_uid}))
                << "exp_id: " << exp_id << " bf8_id: " << bf8_uid << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(bf8_ocp_t{bf8_uid});
        }
    }

    /// Test vector conversions

    auto i = 256 * 256;

    // bf8x2 -> f32x2
    EXPECT_EQ(out[i++], -powf(2.0f, -11.0f));
    EXPECT_EQ(out[i++], powf(2.0f, -13.0f));

    // f32x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);
    // SR
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 1.0f);

    /// Test round to nearest even
    EXPECT_EQ(out[i++], 1024.0f / 4.0f) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];

    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Lowest()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], powf(2.0f, 14.0f)) << "out[i-1]: " << out[i - 1];

    // f32x8 <-> bf8x8 PK8 conversion
    // RNE
    EXPECT_EQ(out[i++], 1.0f);
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 32768.0f);
    EXPECT_EQ(out[i++], -32768.0f);
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()) * 2.f)
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], 0.0f);
    EXPECT_EQ(out[i++], 0.01953125f);
    // SR
    EXPECT_EQ(out[i++], 1.0f);
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 32768.0f);
    EXPECT_EQ(out[i++], -32768.0f);
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()) * 2.f)
        << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(out[i] == 0.0f || out[i] == powf(2.0f, -15.0f));
    i++;
    EXPECT_TRUE(out[i] == 0.01953125f || out[i] == 0.0234375f);
    i++;

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__global__ void test_mx_bf8_device_scaled_convert(uint64_t N, float* p_test, uint64_t* p_completed)
{
    test_mx_bf8_scaled_convert(N, p_test, p_completed);
}

TEST(MXBF8, DeviceScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);

    DeviceMem device_out(test_size * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf8_device_scaled_convert<<<1, 1>>>(
        test_size,
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    // V = X * P; X - E8M0 scale, P - BF8

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        auto idx = e8m0_nan_id * 256 + bf8_id;
        ASSERT_TRUE(std::isnan(out[idx])) << "idx: " << idx << " out[idx]: " << out[idx];
    }

    // If P in {Inf, NaN}, then V = P
    std::set<uint8_t> bf8_spec_ids;
    bf8_spec_ids.insert(0b11111111); //-NaN
    bf8_spec_ids.insert(0b01111111); // +NaN
    bf8_spec_ids.insert(0b11111101); //-NaN
    bf8_spec_ids.insert(0b01111101); // +NaN
    bf8_spec_ids.insert(0b11111110); //-NaN
    bf8_spec_ids.insert(0b01111110); // +NaN
    bf8_spec_ids.insert(0b11111100); //-inf
    bf8_spec_ids.insert(0b01111100); // +inf
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(auto bf8_spec_id : bf8_spec_ids)
        {
            auto idx = exp_id * 256 + bf8_spec_id;

            if(std::isnan(type_convert<float>(bf8_ocp_t{bf8_spec_id})))
            {
                ASSERT_TRUE(std::isnan(out[idx]))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id}) << " != " << out[idx];
            }
            else
            {
                ASSERT_EQ(out[idx], type_convert<float>(bf8_ocp_t{bf8_spec_id}))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id}) << " != " << out[idx];
            }
        }
    }

    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
        {
            if(bf8_spec_ids.find(bf8_id) != bf8_spec_ids.end())
                continue;

            uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
            auto idx        = exp_id * 256 + bf8_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<float>(e8m0_bexp_t(exp_id)) *
                                type_convert<float>(bf8_ocp_t{bf8_uid}))
                << "exp_id: " << exp_id << " bf8_id: " << bf8_uid << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(bf8_ocp_t{bf8_uid});
        }
    }

    /// Test vector conversions

    auto i = 256 * 256;

    // bf8x2 -> f32x2
    EXPECT_EQ(out[i++], -powf(2.0f, -11.0f));
    EXPECT_EQ(out[i++], powf(2.0f, -13.0f));

    // f32x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);
    // SR
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 1.0f);

    /// Test round to nearest even
    EXPECT_EQ(out[i++], 1024.0f / 4.0f) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
#if 1
    EXPECT_TRUE(std::isinf(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isinf(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isinf(out[i++])) << "out[i-1]: " << out[i - 1];
#else
    // NOTE: Host and Device have different behavior.
    // Device returns Infs, while Host returns Max (saturation to finite value).
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Lowest()))
        << "out[i-1]: " << out[i - 1];
#endif
    EXPECT_EQ(out[i++], powf(2.0f, 14.0f)) << "out[i-1]: " << out[i - 1];

    // f32x8 <-> bf8x8 PK8 conversion
    // RNE
    EXPECT_EQ(out[i++], 1.0f);
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 32768.0f);
    EXPECT_EQ(out[i++], -32768.0f);
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isinf(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], 0.0f);
    EXPECT_EQ(out[i++], 0.01953125f);
    // SR
    EXPECT_EQ(out[i++], 1.0f);
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 32768.0f);
    EXPECT_EQ(out[i++], -32768.0f);
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isinf(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(out[i] == 0.0f || out[i] == powf(2.0f, -15.0f));
    i++;
    EXPECT_TRUE(out[i] == 0.01953125f || out[i] == 0.0234375f);
    i++;

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__host__ __device__ float vec16_generator(ck::index_t i) { return powf(-1.0f, i) * powf(2.0f, i); }

__global__ void test_mx_bf8x16_device_scaled_convert(float* p_test, uint64_t* p_completed)
{
    constexpr int N = 16;
    if(p_completed == nullptr)
    {
        return;
    }

    uint64_t& i = *p_completed;
    i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    auto scale2 = e8m0_bexp_t(2.0f);

    bf8x16_ocp_t bf8x16{};
    float16_t float16{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float16[static_cast<int>(ii)] = vec16_generator(ii); });

    bf8x16 = scaled_type_convert<bf8x16_ocp_t>(scale2, float16);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(bf8x16.AsType<bf8_ocp_t>()(ck::Number<ii>{}));
    });
}

TEST(MXBF8, DeviceF32x16ToBF8x16ScaledConvert)
{
    constexpr int N = 16;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf8x16_device_scaled_convert<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i = 0;

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec16_generator(ii) / 2.0f) << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

__host__ __device__ float vec32_generator(ck::index_t i)
{
    if(i < 16)
    {
        return vec16_generator(i % 16);
    }
    else
    {
        return 1.5f * vec16_generator(i % 16);
    }
}

__global__ void test_mx_bf8x32_device_scaled_convert(float* p_test, uint64_t* p_completed)
{
    constexpr int N = 32;
    if(p_completed == nullptr)
    {
        return;
    }

    uint64_t& i = *p_completed;
    i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    auto scale2 = e8m0_bexp_t(2.0f);

    bf8x32_ocp_t bf8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float32[static_cast<int>(ii)] = vec32_generator(ii); });

    bf8x32 = mxf8_convert_rne<bf8x32_ocp_t>(float32, type_convert<float>(scale2));

    ck::static_for<0, N, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(bf8x32.AsType<bf8_ocp_t>()(ck::Number<ii>{}));
    });
}

TEST(MXBF8, DeviceF32x32ToBF8x32ScaledConvert)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf8x32_device_scaled_convert<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i = 0;

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec32_generator(ii) / 2.0f) << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

__global__ void test_mx_bf8x32_device_scaled_convert_sr(float* p_test, uint64_t* p_completed)
{
    constexpr int N = 32;
    if(p_completed == nullptr)
    {
        return;
    }

    uint64_t& i = *p_completed;
    i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    auto scale2 = e8m0_bexp_t(8.0f);

    bf8x32_ocp_t bf8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float32[static_cast<int>(ii)] = vec32_generator(ii); });

    bf8x32 = mxf8_convert_sr<bf8x32_ocp_t>(float32, type_convert<float>(scale2));

    ck::static_for<0, N, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(bf8x32.AsType<bf8_ocp_t>()(ck::Number<ii>{}));
    });
}

TEST(MXBF8, DeviceF32x32ToBF8x32ScaledConvertSR)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf8x32_device_scaled_convert_sr<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i = 0;

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec32_generator(ii) / 8.0f) << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

__global__ void test_mx_f32x32_device_scaled_convert(float* p_test, uint64_t* p_completed)
{
    constexpr int N = 32;
    if(p_completed == nullptr)
    {
        return;
    }

    uint64_t& i = *p_completed;
    i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    auto scale4 = e8m0_bexp_t(4.0f);

    bf8x32_ocp_t bf8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}([&](auto ii) {
        bf8x32.AsType<bf8_ocp_t>()(ii) = type_convert<bf8_ocp_t>(vec32_generator(ii) / 16.0f);
    });

    float32 = scaled_type_convert<float32_t>(scale4, bf8x32);

    ck::static_for<0, N, 1>{}([&](auto ii) { p_test[i++] = float32[static_cast<int>(ii)]; });
}

TEST(MXBF8, DeviceBF8x32ToF32x32ScaledConvert)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_f32x32_device_scaled_convert<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i = 0;

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec32_generator(ii) / 4.0f) << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

// float16
/**
 * @brief Validation for "T(FP16/BF16) convert from all possible combinations of E8M0 and bf8
 * values" Test.
 *
 * @param out T array converted from bf8 values.
 *
 */
template <typename T>
static inline void validate_allcomb(T* out)
{
    // V = X * P; X - E8M0 scale, P - BF8

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        auto idx = e8m0_nan_id * 256 + bf8_id;
        ASSERT_TRUE(std::isnan(type_convert<float>(out[idx])));
    }

    // If P in {Inf, NaN}, then V = P
    std::set<uint8_t> bf8_spec_ids;
    bf8_spec_ids.insert(0b11111111); // -NaN
    bf8_spec_ids.insert(0b01111111); // +NaN
    bf8_spec_ids.insert(0b11111101); // -NaN
    bf8_spec_ids.insert(0b01111101); // +NaN
    bf8_spec_ids.insert(0b11111110); // -NaN
    bf8_spec_ids.insert(0b01111110); // +NaN
    bf8_spec_ids.insert(0b11111100); // -inf
    bf8_spec_ids.insert(0b01111100); // +inf
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(auto bf8_spec_id : bf8_spec_ids)
        {
            auto idx = exp_id * 256 + bf8_spec_id;

            if(std::isnan(type_convert<float>(bf8_ocp_t{bf8_spec_id})))
            {
                ASSERT_TRUE(std::isnan(type_convert<float>(out[idx])))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id})
                    << " != " << type_convert<float>(out[idx]);
            }
            else
            {
                ASSERT_EQ(out[idx], type_convert<T>(bf8_ocp_t{bf8_spec_id}))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id})
                    << " != " << type_convert<float>(out[idx]);
            }
        }
    }

    // V = X * P; X, P - finite
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
        {
            if(bf8_spec_ids.find(bf8_id) != bf8_spec_ids.end())
                continue;

            uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
            auto idx        = exp_id * 256 + bf8_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<T>(type_convert<float>(e8m0_bexp_t(exp_id)) *
                                            type_convert<float>(bf8_ocp_t{bf8_uid})))
                << "exp_id: " << exp_id << " bf8_id: " << bf8_uid << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(bf8_ocp_t{bf8_uid});
        }
    }
}

/**
 * @brief Host version of "T(FP16/BF16) convert from all possible combinations of E8M0 and BF8
 * values".
 *
 * This function performs packed 8 conversions from BF8 values to T values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 (256) and BF8 (256) values.
 * Each thread in a wave holds 8 bf8 values and the wave hold all representative bf8 values.
 *
 * @param Nfp8 The number of bf8 values.
 * @param Nexp The number of exponents
 * @param p_test Pointer to the output array where the converted T values will be stored.
 *
 */
template <typename T>
__host__ void test_host_scaled_convert_allcomb(int Nfp8, int Nexp, T* p_test)
{
    using T8 = typename ck::vector_type<T, 8>::type;
    if(p_test == nullptr)
    {
        return;
    }
    int i = 0;

    // All possible combinations of E8M0 and BF8
    for(ck::index_t exp_id = 0; exp_id < Nexp; exp_id++)
    {
        for(ck::index_t fp8_id = 0; fp8_id < Nfp8; fp8_id += 8)
        {
            bf8x8_ocp_t vf8;
            ck::static_for<0, 8, 1>{}([&](auto ii) {
                vf8.AsType<bf8_ocp_t>()(ck::Number<ii>{}) =
                    bf8_ocp_t{static_cast<uint8_t>(ii + fp8_id)};
            });
            auto vT8 = scaled_type_convert<T8>(e8m0_bexp_t(exp_id), vf8);

            ck::static_for<0, 8, 1>{}([&](auto ii) { p_test[i++] = vT8[static_cast<int>(ii)]; });
            if(i >= Nfp8 * Nexp)
            {
                return;
            }
        }
    }
}

TEST(MXBF8, HostScaledConvertFP16_AllComb)
{
    int test_fp8 = 256;
    int test_exp = 256;
    auto N       = test_fp8 * test_exp;
    std::vector<half_t> out(N, -1.0f);

    test_host_scaled_convert_allcomb(test_fp8, test_exp, out.data());

    validate_allcomb(out.data());
}

/**
 * @brief Device version of "T(FP16/BF16) convert from all possible combinations of E8M0 and BF8
 * values".
 *
 * This function performs packed 8 conversions from BF8 values to T values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 (256) and BF8 (256) values.
 * Each thread in a wave holds 8 bf8 values and the wave hold all representative bf8 values.
 *
 * @param Nfp8 The number of bf8 values.
 * @param Nexp The number of exponents
 * @param p_test Pointer to the output array where the converted T values will be stored.
 *
 */
template <typename T>
__global__ void test_device_scaled_convert_allcomb(int Nfp8, int Nexp, T* p_test)
{
    using T8 = typename ck::vector_type<T, 8>::type;
    if(p_test == nullptr)
    {
        return;
    }

    // All possible combinations of BF8, each thread holds 8
    ck::index_t tid = threadIdx.x;
    bf8x8_ocp_t vf8;
    ck::static_for<0, 8, 1>{}([&](auto ii) {
        vf8.AsType<bf8_ocp_t>()(ck::Number<ii>{}) = bf8_ocp_t{static_cast<uint8_t>(ii + tid * 8)};
    });

    // All possible combinations of E8M0
    T8 vT8;
    for(ck::index_t exp_id = 0; exp_id < Nexp; exp_id++)
    {
        vT8 = scaled_type_convert<T8, bf8x8_ocp_t>(e8m0_bexp_t{exp_id}, vf8);
        ck::static_for<0, 8, 1>{}(
            [&](auto ii) { p_test[ii + tid * 8 + exp_id * Nfp8] = vT8[static_cast<int>(ii)]; });
    }
}

TEST(MXBF8, DeviceScaledConvertFP16_AllComb)
{
    int test_fp8 = 256;
    int test_exp = 256;
    auto N       = test_fp8 * test_exp;
    std::vector<half_t> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(half_t));

    device_out.SetValue(-21.0f);

    test_device_scaled_convert_allcomb<<<1, 32>>>(
        test_fp8, test_exp, static_cast<half_t*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    validate_allcomb(out.data());
}

// All possible BF8 combination test for Bfloat16
TEST(MXBF8, HostScaledConvertBF16_AllComb)
{
    int test_fp8 = 256;
    int test_exp = 256;
    auto N       = test_fp8 * test_exp;
    std::vector<bhalf_t> out(N, -1.0f);

    test_host_scaled_convert_allcomb(test_fp8, test_exp, out.data());

    validate_allcomb(out.data());
}

TEST(MXBF8, DeviceScaledConvertBF16_AllComb)
{
    int test_fp8 = 256;
    int test_exp = 256;
    auto N       = test_fp8 * test_exp;
    std::vector<bhalf_t> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(bhalf_t));

    device_out.SetValue(-21.0f);

    test_device_scaled_convert_allcomb<<<1, 32>>>(
        test_fp8, test_exp, static_cast<bhalf_t*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    validate_allcomb(out.data());
}

/**                                                                                   \
 * @brief Validation for "BF8 to T(FP16/BF16) conversion back and forth" Test.                \
 *                                                                                    \
 * @param out T array converted from bf8 values which converted from a T array. \
 *                                                                                    \
 */
template <typename T>
static inline void validate_2way(uint64_t N, T* out, uint64_t completed, bool device_call)
{
    static_assert(std::is_same_v<T, half_t> || std::is_same_v<T, bhalf_t>,
                  "T must be float16 or bfloat16");
    EXPECT_EQ(N, completed);
    auto i = 0;
    for(int iop = 0; iop < 3; iop++)
    { // single, pk2, pk8 Ops on same test data
        // RNE
        EXPECT_EQ(out[i++], type_convert<T>(1.0f));
        EXPECT_EQ(out[i++], type_convert<T>(-2.0f));
        EXPECT_EQ(out[i++], type_convert<T>(28672.0f));
        EXPECT_TRUE(std::isnan(type_convert<float>(out[i++])));
        if constexpr(std::is_same_v<T, half_t>)
        {
            // f16 max 65504 /4.0f = 16376 -> bf8 16384 * 2 -> bf16 32768
            EXPECT_EQ(out[i++], type_convert<T>(32768.0f));
        }
        else if constexpr(std::is_same_v<T, bhalf_t>)
        {
            if(device_call)
            {
                // device : bf16 max -> bf8 inf * 2 -> bf16 inf
                EXPECT_TRUE(std::isinf(type_convert<float>(out[i++])));
            }
            else
            {
                // host : bf16 max -> bf8 max * 2 -> bf16
                EXPECT_EQ(out[i++],
                          type_convert<T>(type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()) *
                                          2.f));
            }
        }
        EXPECT_EQ(out[i++], type_convert<T>(powf(2.0f, -15.0f)));
        EXPECT_EQ(out[i++], type_convert<T>(-powf(2.0f, -13.0f)));
        EXPECT_EQ(out[i++], type_convert<T>(0.0f));
        // SR
        EXPECT_EQ(out[i++], type_convert<T>(1.0f));
        EXPECT_EQ(out[i++], type_convert<T>(-2.0f));
        EXPECT_EQ(out[i++], type_convert<T>(28672.0f));
        EXPECT_TRUE(std::isnan(type_convert<float>(out[i++])));
        if constexpr(std::is_same_v<T, half_t>)
        {
            // f16 max 65504 /4.0f = 16376 in [14336, 16384] bf8 * 2 -> bf16 [28672, 32768]
            EXPECT_TRUE(ck::bit_cast<uint16_t>(out[i]) ==
                            ck::bit_cast<uint16_t>(type_convert<T>(28672.0f)) ||
                        ck::bit_cast<uint16_t>(out[i]) ==
                            ck::bit_cast<uint16_t>(type_convert<T>(32768.0f)));
            i++;
        }
        else if constexpr(std::is_same_v<T, bhalf_t>)
        {
            if(device_call)
            {
                // device : bf16 max -> bf8 inf * 2 -> bf16 inf
                EXPECT_TRUE(std::isinf(type_convert<float>(out[i++])));
            }
            else
            {
                // host : bf16 max -> bf8 max * 2 -> bf16
                EXPECT_EQ(out[i++],
                          type_convert<T>(type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()) *
                                          2.f));
            }
        }
        EXPECT_EQ(out[i++], type_convert<T>(powf(2.0f, -15.0f)));
        EXPECT_EQ(out[i++], type_convert<T>(-powf(2.0f, -13.0f)));
        EXPECT_TRUE(ck::bit_cast<uint16_t>(out[i]) == uint16_t{0x0000} ||
                    ck::bit_cast<uint16_t>(out[i]) ==
                        ck::bit_cast<uint16_t>(type_convert<T>(powf(2.0f, -15.0f))));
        i++;
    }
    EXPECT_EQ(N, i);
}
/**
 * @brief Device version of "BF8 to T(FP16/BF16) conversion back and forth".
 *
 * This function performs packed 8/ pakced 2/ single value scale conversions from T values to BF8
 * values and back. Both RNE and SR tested.
 *
 * @param N number of value tested.
 * @param p_test Pointer to the output array where the converted float16 values will be stored.
 * @param p_completed Pointer to a variable that tracks the number of completed conversions.
 *
 */
template <typename T>
__host__ __device__ void test_pk8_scaled_convert(uint64_t N, T* p_test, uint64_t* p_completed)
{
    using T8 = typename ck::vector_type<T, 8>::type;
    using T2 = typename ck::vector_type<T, 2>::type;
    if(p_completed == nullptr)
    {
        return;
    }

    uint64_t& i = *p_completed;
    i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    // test pk8 vector conversion, first /4.0f, then *2.0f
    float fscale4 = 4.0f;
    auto scale2   = e8m0_bexp_t(2.0f);

    T v_qnan = ck::NumericLimits<T>::QuietNaN();
    T v_max  = ck::NumericLimits<T>::Max();
#if !CK_USE_LLVM_BUILTIN_BF16
    if constexpr(std::is_same_v<T, bhalf_t>)
    {
        v_qnan = bhalf_t{0x7FFF};
        v_max  = bhalf_t{0x7F7F};
    }
#endif
    union
    {
        T8 v8;
        T2 v2[4];
        T v[8];
    } test_set{T8{
        type_convert<T>(2.0f),                                // a positive representable
        type_convert<T>(-4.0f),                               // a negative representable
        type_convert<T>(ck::NumericLimits<bf8_ocp_t>::Max()), // 57344 /4.0f = 14336 (0x73)
                                                              //       *2.0f = 28672
        v_qnan,
        v_max,
        type_convert<T>(powf(2.0f, -14.0f)),  // 2^-16 smallest positive subnorm in bf8
        type_convert<T>(-powf(2.0f, -12.0f)), // -2^-14 smallest negative norm in bf8
        type_convert<T>(powf(2.0f, -16.0f))   // 2^-18 not representable in bf8
    }};

    // packed 8 Ops
    // rne
    auto v8_back =
        scaled_type_convert<T8>(scale2, mxf8_convert_rne<bf8x8_ocp_t>(test_set.v8, fscale4));

    ck::static_for<0, 8, 1>{}([&](auto ii) { p_test[i++] = v8_back[static_cast<int>(ii)]; });
    if(i >= N)
    {
        return;
    }

    // sr
    v8_back = scaled_type_convert<T8>(scale2, mxf8_convert_sr<bf8x8_ocp_t>(test_set.v8, fscale4));

    ck::static_for<0, 8, 1>{}([&](auto ii) { p_test[i++] = v8_back[static_cast<int>(ii)]; });
    if(i >= N)
    {
        return;
    }

    // packed 2 Ops
    T2 v2_back[4];
    ck::static_for<0, 4, 1>{}([&](auto ii) {
        v2_back[ii] = scaled_type_convert<T2>(
            scale2, mxf8_convert_rne<bf8x2_ocp_t>(test_set.v2[ck::Number<ii>{}], fscale4));
        p_test[i++] = v2_back[ii][0];
        p_test[i++] = v2_back[ii][1];
        if(i >= N)
        {
            return;
        }
    });

    // sr
    ck::static_for<0, 4, 1>{}([&](auto ii) {
        v2_back[ii] = scaled_type_convert<T2>(
            scale2, mxf8_convert_sr<bf8x2_ocp_t>(test_set.v2[ck::Number<ii>{}], fscale4));
        p_test[i++] = v2_back[ii][0];
        p_test[i++] = v2_back[ii][1];
        if(i >= N)
        {
            return;
        }
    });

    // single value Ops
    T v_back[8];
    ck::static_for<0, 8, 1>{}([&](auto ii) {
        v_back[ii] = scaled_type_convert<T>(
            scale2, mxf8_convert_rne<bf8_ocp_t>(test_set.v[ck::Number<ii>{}], fscale4));
        p_test[i++] = v_back[ii];
        if(i >= N)
        {
            return;
        }
    });

    // sr
    ck::static_for<0, 8, 1>{}([&](auto ii) {
        v_back[ii] = scaled_type_convert<T>(
            scale2, mxf8_convert_sr<bf8_ocp_t>(test_set.v[ck::Number<ii>{}], fscale4));
        p_test[i++] = v_back[ii];
        if(i >= N)
        {
            return;
        }
    });
}

TEST(MXBF8, HostF16x8_BF8x8ScaledConvert)
{
    constexpr uint64_t N = 8 * 2 * 3; // test 8 values for RNE and SR with single, pk2, pk8 cvt Ops
    uint64_t completed   = 0;
    std::vector<half_t> out(N, -1.0f);

    test_pk8_scaled_convert(N, out.data(), &completed);

    validate_2way(N, out.data(), completed, false);
}

template <typename T>
__global__ void test_device_pk8_scaled_convert(uint64_t N, T* p_test, uint64_t* p_completed)
{
    test_pk8_scaled_convert(N, p_test, p_completed);
}

TEST(MXBF8, DeviceF16x8_BF8x8ScaledConvert)
{
    constexpr uint64_t N = 8 * 2 * 3; // test 8 values for RNE and SR with single, pk2, pk8 cvt Ops
    std::vector<half_t> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(half_t));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_device_pk8_scaled_convert<<<1, 1>>>(
        N,
        static_cast<half_t*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    validate_2way(N, out.data(), completed, true);
}

// bfloat16
TEST(MXBF8, HostBF16x8_BF8x8ScaledConvert)
{
    constexpr uint64_t N = 8 * 2 * 3; // test 8 values for RNE and SR with single, pk2, pk8 cvt Ops
    uint64_t completed   = 0;
    std::vector<bhalf_t> out(N, -1.0f);

    test_pk8_scaled_convert(N, out.data(), &completed);

    validate_2way(N, out.data(), completed, false);
}

TEST(MXBF8, DeviceBF16x8_BF8x8ScaledConvert)
{
    constexpr uint64_t N = 8 * 2 * 3; // test 8 values for RNE and SR with single, pk2, pk8 cvt Ops
    std::vector<bhalf_t> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(bhalf_t));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_device_pk8_scaled_convert<<<1, 1>>>(
        N,
        static_cast<bhalf_t*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    validate_2way(N, out.data(), completed, true);
}
