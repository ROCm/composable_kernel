// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bf6_convert_rne;
using ck::bf6_convert_sr;
using ck::bf6_t;
using ck::bf6x16_pk_t;
using ck::bf6x16_t;
using ck::bf6x32_pk_t;
using ck::bf6x32_t;
using ck::bhalf_t;
using ck::DeviceMem;
using ck::e8m0_bexp_t;
using ck::half_t;
using ck::scaled_type_convert;
using ck::type_convert;

// Test size: 256 E8M0 values * 64 BF6 values + vector tests + edge cases
constexpr uint64_t test_size = 256 * 64 + (16 + 32) * 3 + 8;

template <typename T>
class MXBF6TypedTest : public ::testing::Test
{
};
using TestTypes = ::testing::Types<float, half_t, bhalf_t>;
TYPED_TEST_SUITE(MXBF6TypedTest, TestTypes);

/* generate float values from representative bf values
   positive values and sample even bit pattern */
__host__ __device__ float vec16_generator(ck::index_t i, float scale)
{
    return scale * type_convert<float>((bf6_t((i * 2) & 0b00111111)));
}

/* generate float values from representative bf values
   positive and negative values and sample odd values */
__host__ __device__ float vec32_generator(ck::index_t i, float scale)
{

    return scale * type_convert<float>((bf6_t((i * 2 + 1) & 0b00111111)));
}
/**
 * @brief Tests conversion of BF6 values to T(float/half_t/bhalf_t) using E8M0 exponent scaling.
 *
 * This function performs a series of conversions from BF6 values to T values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 and BF6 values,
 * as well as specific vector and rounding conversions.
 *
 * @param N The maximum number of conversions to perform.
 * @param p_test Pointer to the output array where the converted T values will be stored.
 * @param p_completed Pointer to a variable that tracks the number of completed conversions.
 *
 * @note First 256*64 conversions are for all possible combinations of E8M0 and BF6 values
 * stored sequentially with BF6 values varying faster.
 *
 * The function performs the following conversions:
 * - All possible combinations of E8M0 and BF6 values. [256x64]
 * - Vector conversions f6x16 -> Tx16. [16]
 * - Vector conversions f6x32 -> Tx32. [32]
 * - Vector conversions Tx16 -> f6x16 rne. [16]
 * - Vector conversions Tx32 -> f6x32 rne. [32]
 * - Vector conversions Tx16 -> f6x16 sr. [16]
 * - Vector conversions Tx32 -> f6x32 sr. [32]
 * - Round to nearest even conversions for specific T values. [8]
 */
template <typename T>
__host__ __device__ void test_mx_bf6_scaled_convert(uint64_t N, T* p_test, uint64_t* p_completed)
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

    // All possible combinations of E8M0 and bf6
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        for(ck::index_t fp6_id = 0; fp6_id < 64; fp6_id++)
        {
            uint8_t fp6_uid = static_cast<uint8_t>(fp6_id);
            auto v    = scaled_type_convert<T>(e8m0_bexp_t(exp_id), bf6_t(fp6_uid & 0b00111111));
            p_test[i] = v;
            i++;
            if(i >= N)
            {
                return;
            }
        }
    }

    /// Test vector conversions
    // bf6x16 -> Tx16
    auto scale2 = e8m0_bexp_t(2.0f);

    // Create f6x16 with specific pattern
    bf6x16_pk_t bf6x16_pk{};
    for(ck::index_t j = 0; j < 16; j++)
    {
        bf6x16_pk.pack(bf6_t((j * 2) & 0b00111111), j);
    }

    bf6x16_t f6x16 = bf6x16_pk;
    T16 vTx16      = scaled_type_convert<T16>(scale2, f6x16);
    ck::static_for<0, 16, 1>{}([&](auto j) { p_test[i++] = vTx16[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // bf6x32 -> Tx32
    bf6x32_pk_t bf6x32_pk{};
    for(ck::index_t j = 0; j < 32; j++)
    {
        bf6x32_pk.pack(bf6_t((j) & 0b00111111), j);
    }
    bf6x32_t f6x32 = bf6x32_pk;

    T32 vTx32 = scaled_type_convert<T32>(scale2, f6x32);
    ck::static_for<0, 32, 1>{}([&](auto j) { p_test[i++] = vTx32[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // Tx16 -> f6x16 RNE
    T16 test_vec16{};
    for(int j = 0; j < 16; j++)
    {
        test_vec16[j] = type_convert<T>(vec16_generator(j, type_convert<float>(scale2)));
    }
    f6x16           = bf6_convert_rne(test_vec16, type_convert<float>(scale2));
    auto vTx16_back = type_convert<T16>(f6x16);
    ck::static_for<0, 16, 1>{}([&](auto j) { p_test[i++] = vTx16_back[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // Tx32 -> f6x32 RNE
    T32 test_vec32{};
    for(int j = 0; j < 32; j++)
    {
        test_vec32[j] = type_convert<T>(vec32_generator(j, type_convert<float>(scale2)));
    }
    f6x32           = bf6_convert_rne(test_vec32, type_convert<float>(scale2));
    auto vTx32_back = type_convert<T32>(f6x32);
    ck::static_for<0, 32, 1>{}([&](auto j) { p_test[i++] = vTx32_back[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // Tx16 -> f6x16 SR
    f6x16      = bf6_convert_sr(test_vec16, type_convert<float>(scale2));
    vTx16_back = type_convert<T16>(f6x16);
    ck::static_for<0, 16, 1>{}([&](auto j) { p_test[i++] = vTx16_back[static_cast<int>(j)]; });
    if(i >= N)
        return;

    // Tx32 -> f6x32 SR
    f6x32      = bf6_convert_sr(test_vec32, type_convert<float>(scale2));
    vTx32_back = type_convert<T32>(f6x32);
    ck::static_for<0, 32, 1>{}([&](auto j) { p_test[i++] = vTx32_back[static_cast<int>(j)]; });
    if(i >= N)
        return;

    /// Test round to nearest even with scaling: 4.75/4 = 1.1875 (1.124 and 1.25), RNE pick 1.25
    p_test[i++] = type_convert<T>(bf6_convert_rne(type_convert<T>(4.75f), 4.0f));
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
    p_test[i++] = type_convert<T>(bf6_convert_rne(v_qnan, 4.0f));
    if(i >= N)
        return;

    // Inf/2 > 7.5 => saturate to 7.5
    p_test[i++] = type_convert<T>(bf6_convert_rne(v_infinity, 2.0f));
    if(i >= N)
        return;

    // 512/0.5 > 7.5 => saturate to 7.5
    p_test[i++] = type_convert<T>(bf6_convert_rne(type_convert<T>(512.0f), 0.5f));
    if(i >= N)
        return;

    // -512/0.5 < -7.5 => saturate to -7.5
    p_test[i++] = type_convert<T>(bf6_convert_rne(type_convert<T>(-512.0f), 0.5f));
    if(i >= N)
        return;

    // Test proper scale: 14.0/2.0 = 7.0
    p_test[i++] = type_convert<T>(bf6_convert_rne(type_convert<T>(14.0f), 2.0f));
    if(i >= N)
        return;

    // Test subnormal: 0.25/2.0 = 0.125
    p_test[i++] = type_convert<T>(bf6_convert_rne(type_convert<T>(0.25f), 2.0f));
    if(i >= N)
        return;

    // Test zero
    p_test[i++] = type_convert<T>(bf6_convert_rne(type_convert<T>(0.0f), 1.0f));
    if(i >= N)
        return;
}

template <typename T>
static inline void validate(T* out)
{
    // V = X * P; X - E8M0 scale, P - bf6

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
                                            type_convert<float>(bf6_t(fp6_uid & 0b00111111))))
                << "exp_id: " << exp_id << " fp6_id: " << fp6_id << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(bf6_t(fp6_uid & 0b00111111));
        }
    }

    /// Test vector conversions
    auto i = 256 * 64;

    // f6x16 -> Tx16: validate all 16 elements
    constexpr float f6x16_expected[16] = {
        0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0};
    for(int j = 0; j < 16; j++)
    {
        EXPECT_EQ(out[i++], type_convert<T>(f6x16_expected[j]))
            << "f6x16[" << j << "] -> Tx16 conversion failed";
    }

    // f6x32 -> Tx32: validate all 32 elements
    constexpr float f6x32_expected[32] = {0.0,  0.125, 0.25, 0.375, 0.5,  0.625, 0.75, 0.875,
                                          1.0,  1.25,  1.5,  1.75,  2.0,  2.5,   3.0,  3.5,
                                          4.0,  5.0,   6.0,  7.0,   8.0,  10.0,  12.0, 14.0,
                                          16.0, 20.0,  24.0, 28.0,  32.0, 40.0,  48.0, 56.0};
    for(int j = 0; j < 32; j++)
    {
        EXPECT_EQ(out[i++], type_convert<T>(f6x32_expected[j]))
            << "f6x32[" << j << "] -> Tx32 conversion failed";
    }

    // Tx16 -> f6x16 RNE: validate all 16 elements
    for(int j = 0; j < 16; j++)
    {
        float input    = vec16_generator(j, 2.0f);
        float expected = input / 2.0f; // After scale division
        if(expected > 28.0f)
            expected = 28.0f;
        EXPECT_EQ(out[i++], type_convert<T>(expected))
            << "Tx16[" << j << "] -> f6x16 RNE conversion failed, input=" << input;
    }

    // Tx32 -> f6x32 RNE: validate all 32 elements
    for(int j = 0; j < 32; j++)
    {
        float input    = vec32_generator(j, 2.0f);
        float expected = input / 2.0f; // After scale division
        // Values outside [-7.5, 7.5] saturate
        if(expected > 28.0f)
            expected = 28.0f;
        if(expected < -28.0f)
            expected = -28.0f;
        EXPECT_EQ(out[i++], type_convert<T>(expected))
            << "Tx32[" << j << "] -> f6x32 RNE conversion failed, input=" << input;
    }

    // Tx16 -> f6x16 SR: validate all 16 elements
    for(int j = 0; j < 16; j++)
    {
        float input    = vec16_generator(j, 2.0f);
        float expected = input / 2.0f; // After scale division
        if(expected > 28.0f)
            expected = 28.0f;
        EXPECT_EQ(out[i++], type_convert<T>(expected))
            << "Tx16[" << j << "] -> f6x16 SR conversion failed, input=" << input;
    }

    // Tx32 -> f6x32 SR: validate all 32 elements
    for(int j = 0; j < 32; j++)
    {
        float input    = vec32_generator(j, 2.0f);
        float expected = input / 2.0f; // After scale division
        // Values outside [-7.5, 7.5] saturate
        if(expected > 28.0f)
            expected = 28.0f;
        if(expected < -28.0f)
            expected = -28.0f;
        EXPECT_EQ(out[i++], type_convert<T>(expected))
            << "Tx32[" << j << "] -> f6x32 SR conversion failed, input=" << input;
    }

    /// Test round to nearest even: 4.75/4 = 1.1875 -> RNE picks 1.25 (even mantissa)
    EXPECT_EQ(out[i++], type_convert<T>(1.25f)) << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<bf6_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<bf6_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<bf6_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<bf6_t>::Lowest()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(7.0f)) << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(0.125f)) << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(0.0f)) << "out[i-1]: " << type_convert<float>(out[i - 1]);

    EXPECT_EQ(test_size, i);
}

TYPED_TEST(MXBF6TypedTest, HostScaledConvert)
{
    using T = TypeParam;
    std::vector<T> out(test_size, T(-1.0f));
    uint64_t completed = 0;

    test_mx_bf6_scaled_convert(test_size, out.data(), &completed);

    EXPECT_EQ(test_size, completed);
    validate(out.data());
}

template <typename T>
__global__ void test_mx_bf6_device_scaled_convert(uint64_t N, T* p_test, uint64_t* p_completed)
{
    test_mx_bf6_scaled_convert(N, p_test, p_completed);
}

TYPED_TEST(MXBF6TypedTest, DeviceScaledConvert)
{
    using T = TypeParam;
    std::vector<T> out(test_size, T(-1.0f));

    DeviceMem device_out(test_size * sizeof(T));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf6_device_scaled_convert<<<1, 1>>>(
        test_size,
        static_cast<T*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    EXPECT_EQ(test_size, completed);
    validate(out.data());
}
