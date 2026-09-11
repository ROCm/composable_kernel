// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bhalf_t;
using ck::DeviceMem;
using ck::e8m0_bexp_t;
using ck::f6_convert_rne;
using ck::f6_convert_sr;
using ck::f6_t;
using ck::f6x16_pk_t;
using ck::f6x16_t;
using ck::f6x32_pk_t;
using ck::f6x32_t;
using ck::half_t;
using ck::scaled_type_convert;
using ck::type_convert;

// Test size: 256 E8M0 values * 64 FP6 values + vector tests + edge cases
constexpr uint64_t test_size = 256 * 64 + (16 + 32) * 3 + 8;

template <typename T>
class MXFP6TypedTest : public ::testing::Test
{
};
using TestTypes = ::testing::Types<float, half_t, bhalf_t>;
TYPED_TEST_SUITE(MXFP6TypedTest, TestTypes);

/**
 * @brief Tests conversion of FP6 values to T(float/half_t/bhalf_t) using E8M0 exponent scaling.
 *
 * This function performs a series of conversions from FP6 values to T values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 and FP6 values,
 * as well as specific vector and rounding conversions.
 *
 * @param N The maximum number of conversions to perform.
 * @param p_test Pointer to the output array where the converted T values will be stored.
 * @param p_completed Pointer to a variable that tracks the number of completed conversions.
 *
 * @note First 256*64 conversions are for all possible combinations of E8M0 and FP6 values
 * stored sequentially with FP6 values varying faster.
 *
 * The function performs the following conversions:
 * - All possible combinations of E8M0 and FP6 values. [256x64]
 * - Vector conversions f6x16 -> Tx16. [16]
 * - Vector conversions f6x32 -> Tx32. [32]
 * - Vector conversions Tx16 -> f6x16 rne. [16]
 * - Vector conversions Tx32 -> f6x32 rne. [32]
 * - Vector conversions Tx16 -> f6x16 sr. [16]
 * - Vector conversions Tx32 -> f6x32 sr. [32]
 * - Round to nearest even conversions for specific T values. [8]
 */
template <typename T>
__host__ __device__ void test_mx_fp6_scaled_convert(uint64_t N, T* p_test, uint64_t* p_completed)
{
    using T16 = typename ck::vector_type<T, 16>::type;
    using T32 = typename ck::vector_type<T, 32>::type;

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

    // All possible combinations of E8M0 and FP6
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        for(ck::index_t fp6_id = 0; fp6_id < 64; fp6_id++)
        {
            uint8_t fp6_uid = static_cast<uint8_t>(fp6_id);
            auto v    = scaled_type_convert<T>(e8m0_bexp_t(exp_id), f6_t(fp6_uid & 0b00111111));
            p_test[i] = v;
            i++;
            if(i >= N)
            {
                return;
            }
        }
    }

    /// Test vector conversions
    // f6x16 -> Tx16
    auto scale2 = e8m0_bexp_t(2.0f);

    // Create f6x16 with specific pattern
    f6x16_pk_t f6x16_pk{};
    for(ck::index_t j = 0; j < 16; j++)
    {
        f6x16_pk.pack(f6_t((j * 2) & 0b00111111), j);
    }

    f6x16_t f6x16 = f6x16_pk;
    T16 vTx16     = scaled_type_convert<T16>(scale2, f6x16);
    ck::static_for<0, 16, 1>{}([&](auto j) { p_test[i++] = vTx16[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // f6x32 -> Tx32
    f6x32_pk_t f6x32_pk{};
    for(ck::index_t j = 0; j < 32; j++)
    {
        f6x32_pk.pack(f6_t((j) & 0b00111111), j);
    }
    f6x32_t f6x32 = f6x32_pk;

    T32 vTx32 = scaled_type_convert<T32>(scale2, f6x32);
    ck::static_for<0, 32, 1>{}([&](auto j) { p_test[i++] = vTx32[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // Tx16 -> f6x16 RNE
    T16 test_vec16{};
    for(int j = 0; j < 16; j++)
    {
        test_vec16[j] = type_convert<T>((j + 1.0f) * 2.0f);
    }
    f6x16           = f6_convert_rne(test_vec16, type_convert<float>(scale2));
    auto vTx16_back = type_convert<T16>(f6x16);
    ck::static_for<0, 16, 1>{}([&](auto j) { p_test[i++] = vTx16_back[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // Tx32 -> f6x32 RNE
    T32 test_vec32{};
    for(int j = 0; j < 32; j++)
    {
        test_vec32[j] = type_convert<T>((j - 15.5f) * 2.0f);
    }
    f6x32           = f6_convert_rne(test_vec32, type_convert<float>(scale2));
    auto vTx32_back = type_convert<T32>(f6x32);
    ck::static_for<0, 32, 1>{}([&](auto j) { p_test[i++] = vTx32_back[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // Tx16 -> f6x16 SR
    f6x16      = f6_convert_sr(test_vec16, type_convert<float>(scale2));
    vTx16_back = type_convert<T16>(f6x16);
    ck::static_for<0, 16, 1>{}([&](auto j) { p_test[i++] = vTx16_back[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // Tx32 -> f6x32 SR
    f6x32      = f6_convert_sr(test_vec32, type_convert<float>(scale2));
    vTx32_back = type_convert<T32>(f6x32);
    ck::static_for<0, 32, 1>{}([&](auto j) { p_test[i++] = vTx32_back[static_cast<int>(j)]; });
    if(i >= N)
        return;

    /// Test round to nearest even with scaling: 4.75/4 = 1.1875 (1.124 and 1.25), RNE pick 1.25
    p_test[i++] = type_convert<T>(f6_convert_rne(type_convert<T>(4.75f), 4.0f));
    if(i >= N)
        return;

    T v_qnan     = ck::NumericLimits<T>::QuietNaN();
    T v_infinity = ck::NumericLimits<T>::Infinity();
#if !CK_USE_LLVM_BUILTIN_BF16
    if constexpr(std::is_same_v<T, bhalf_t>)
    {
        v_qnan     = bhalf_t{0x7FFF};
        v_infinity = bhalf_t{0x7F80};
    }
#endif

    // NaN -> saturate to max
    p_test[i++] = type_convert<T>(f6_convert_rne(v_qnan, 4.0f));
    if(i >= N)
        return;

    // Inf/2 > 7.5 => saturate to 7.5
    p_test[i++] = type_convert<T>(f6_convert_rne(v_infinity, 2.0f));
    if(i >= N)
        return;

    // 512/0.5 > 7.5 => saturate to 7.5
    p_test[i++] = type_convert<T>(f6_convert_rne(type_convert<T>(512.0f), 0.5f));
    if(i >= N)
        return;

    // -512/0.5 < -7.5 => saturate to -7.5
    p_test[i++] = type_convert<T>(f6_convert_rne(type_convert<T>(-512.0f), 0.5f));
    if(i >= N)
        return;

    // Test proper scale: 14.0/2.0 = 7.0
    p_test[i++] = type_convert<T>(f6_convert_rne(type_convert<T>(14.0f), 2.0f));
    if(i >= N)
        return;

    // Test subnormal: 0.25/2.0 = 0.125
    p_test[i++] = type_convert<T>(f6_convert_rne(type_convert<T>(0.25f), 2.0f));
    if(i >= N)
        return;

    // Test zero
    p_test[i++] = type_convert<T>(f6_convert_rne(type_convert<T>(0.0f), 1.0f));
    if(i >= N)
        return;
}

template <typename T>
static inline void validate(T* out)
{
    // V = X * P; X - E8M0 scale, P - FP6

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t fp6_id = 0; fp6_id < 64; fp6_id++)
    {
        auto idx = e8m0_nan_id * 64 + fp6_id;
        ASSERT_TRUE(std::isnan(type_convert<float>(out[idx])));
    }

    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t fp6_id = 0; fp6_id < 64; fp6_id++)
        {
            uint8_t fp6_uid = static_cast<uint8_t>(fp6_id);
            auto idx        = exp_id * 64 + fp6_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<T>(type_convert<float>(e8m0_bexp_t(exp_id)) *
                                            type_convert<float>(f6_t(fp6_uid & 0b00111111))))
                << "exp_id: " << exp_id << " fp6_id: " << fp6_id << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(f6_t(fp6_uid & 0b00111111));
        }
    }

    /// Test vector conversions
    auto i = 256 * 64;

    // f6x16 -> Tx16: validate all 16 elements
    constexpr float f6x16_expected[16] = {
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0};
    for(int j = 0; j < 16; j++)
    {
        EXPECT_EQ(out[i++], type_convert<T>(f6x16_expected[j]))
            << "f6x16[" << j << "] -> Tx16 conversion failed";
    }

    // f6x32 -> Tx32: validate all 32 elements
    constexpr float f6x32_expected[32] = {0.0f, 0.25f, 0.5f,  0.75f, 1.0f,  1.25f, 1.5f,  1.75f,
                                          2.0f, 2.25f, 2.5f,  2.75f, 3.0f,  3.25f, 3.5f,  3.75f,
                                          4.0f, 4.5f,  5.0f,  5.5f,  6.0f,  6.5f,  7.0f,  7.5f,
                                          8.0f, 9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    for(int j = 0; j < 32; j++)
    {
        EXPECT_EQ(out[i++], type_convert<T>(f6x32_expected[j]))
            << "f6x32[" << j << "] -> Tx32 conversion failed";
    }

    // Tx16 -> f6x16 RNE: validate all 16 elements
    for(int j = 0; j < 16; j++)
    {
        float input    = (j + 1.0f) * 2.0f;
        float expected = input / 2.0f; // After scale division
        // Values > 7.5 saturate to 7.5
        if(expected > 7.5f)
            expected = 7.5f;
        EXPECT_EQ(out[i++], type_convert<T>(expected))
            << "Tx16[" << j << "] -> f6x16 RNE conversion failed, input=" << input;
    }

    // Tx32 -> f6x32 RNE: validate all 32 elements
    for(int j = 0; j < 32; j++)
    {
        float input    = (j - 15.5f) * 2.0f;
        float expected = input / 2.0f; // After scale division
        // Values outside [-7.5, 7.5] saturate
        if(expected > 7.5f)
            expected = 7.5f;
        if(expected < -7.5f)
            expected = -7.5f;
        EXPECT_EQ(out[i++], type_convert<T>(expected))
            << "Tx32[" << j << "] -> f6x32 RNE conversion failed, input=" << input;
    }

    // Tx16 -> f6x16 SR: validate all 16 elements
    for(int j = 0; j < 16; j++)
    {
        float input    = (j + 1.0f) * 2.0f;
        float expected = input / 2.0f; // After scale division
        // Values > 7.5 saturate to 7.5
        if(expected > 7.5f)
            expected = 7.5f;
        EXPECT_EQ(out[i++], type_convert<T>(expected))
            << "Tx16[" << j << "] -> f6x16 SR conversion failed, input=" << input;
    }

    // Tx32 -> f6x32 SR: validate all 32 elements
    for(int j = 0; j < 32; j++)
    {
        float input    = (j - 15.5f) * 2.0f;
        float expected = input / 2.0f; // After scale division
        // Values outside [-7.5, 7.5] saturate
        if(expected > 7.5f)
            expected = 7.5f;
        if(expected < -7.5f)
            expected = -7.5f;
        EXPECT_EQ(out[i++], type_convert<T>(expected))
            << "Tx32[" << j << "] -> f6x32 SR conversion failed, input=" << input;
    }

    /// Test round to nearest even: 4.75/4 = 1.1875 -> RNE picks 1.25 (even mantissa)
    EXPECT_EQ(out[i++], type_convert<T>(1.25f)) << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<f6_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<f6_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<f6_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<f6_t>::Lowest()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(7.0f)) << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(0.125f)) << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(0.0f)) << "out[i-1]: " << type_convert<float>(out[i - 1]);

    EXPECT_EQ(test_size, i);
}

TYPED_TEST(MXFP6TypedTest, HostScaledConvert)
{
    using T = TypeParam;
    std::vector<T> out(test_size, T(-1.0f));
    uint64_t completed = 0;

    test_mx_fp6_scaled_convert(test_size, out.data(), &completed);

    EXPECT_EQ(test_size, completed);
    validate(out.data());
}

template <typename T>
__global__ void test_mx_fp6_device_scaled_convert(uint64_t N, T* p_test, uint64_t* p_completed)
{
    test_mx_fp6_scaled_convert(N, p_test, p_completed);
}

TYPED_TEST(MXFP6TypedTest, DeviceScaledConvert)
{
    using T = TypeParam;
    std::vector<T> out(test_size, T(-1.0f));

    DeviceMem device_out(test_size * sizeof(T));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp6_device_scaled_convert<<<1, 1>>>(
        test_size,
        static_cast<T*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    EXPECT_EQ(test_size, completed);
    validate(out.data());
}
