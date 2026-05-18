// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::bhalf2_t;
using ck::bhalf32_t;
using ck::bhalf_t;
using ::ck::DeviceMem;
using ck::e8m0_bexp_t;
using ck::float16_t;
using ck::float2_t;
using ck::float32_t;
using ck::half2_t;
using ck::half32_t;
using ck::half_t;
using ck::scaled_type_convert;
using ck::type_convert;

using ck::f4_convert_rne;
using ck::f4_convert_sr;
using ck::f4_t;
using ck::f4x16_t;
using ck::f4x2_pk_t;
using ck::f4x2_t;
using ck::f4x32_t;
using ck::f4x8_t;

constexpr uint64_t test_size = 256 * 16 + 2 + 4 + 6;

template <typename T>
class MXFP4TypedTest : public ::testing::Test
{
};
using TestTypes = ::testing::Types<float, half_t, bhalf_t>;
TYPED_TEST_SUITE(MXFP4TypedTest, TestTypes);

/**
 * @brief Tests conversion of FP4 values to T(float/float16/bfloat16) using E8M0 exponent scaling.
 *
 * This function performs a series of conversions from FP4 values to T values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 and FP4 values,
 * as well as specific vector and rounding conversions.
 *
 * @param N The maximum number of conversions to perform.
 * @param p_test Pointer to the output array where the converted T values will be stored.
 * @param p_completed Pointer to a variable that tracks the number of completed conversions.
 *
 * @note If either p_test or p_completed is nullptr, the function will return immediately.
 * @note The function will stop converting if the number of conversions reaches N.
 * @note First 256*16 conversions are for all possible combinations of E8M0 and FP4 values that are
 * stored in memory sequentially with FP4 values varying faster.
 *
 * The function performs the following conversions:
 * - All possible combinations of E8M0 and FP4 values. [256x16]
 * - Vector conversions f4x2 -> Tx2. [2]
 * - Vector conversions  Tx2 -> f4x2 rne. [2]
 * - Vector conversions  Tx2 -> f4x2 sr. [2]
 * - Round to nearest even conversions for specific T values. [6]
 *
 * The results are stored in the p_test array, and the number of completed conversions
 * is updated in the p_completed variable.
 */
template <typename T>
__host__ __device__ void test_mx_fp4_scaled_convert(uint64_t N, T* p_test, uint64_t* p_completed)
{
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

    // All possible combinations of E8M0 and FP4
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        for(ck::index_t fp4_id = 0; fp4_id < 16; fp4_id++)
        {
            uint8_t fp4_uid = static_cast<uint8_t>(fp4_id);
            auto v    = scaled_type_convert<T>(e8m0_bexp_t(exp_id), f4_t(fp4_uid & 0b00001111));
            p_test[i] = v;
            i++;
            if(i >= N)
            {
                return;
            }
        }
    }

    /// Test vector conversions
    // f4x2 -> Tx2
    f4x2_t f4x2{f4x2_t::data_v{0b00011100}}; // 0b0001(=0.5) and 0b1100(=-2.0)
    auto scale2 = e8m0_bexp_t(2.0f);

    T2 vTx2     = scaled_type_convert<T2>(scale2, f4x2);
    p_test[i++] = vTx2[0];
    if(i >= N)
    {
        return;
    }
    p_test[i++] = vTx2[1];
    if(i >= N)
    {
        return;
    }

    // Tx2 -> f4x2
    vTx2 = {type_convert<T>(1.0f), type_convert<T>(-4.0f)};
    f4x2 = f4_convert_rne(vTx2, type_convert<float>(scale2)); // expect {0.5, -2}

    p_test[i++] = type_convert<T>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<0>{}))); // 0.5f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<T>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<1>{}))); // -2.0f
    if(i >= N)
    {
        return;
    }

    auto vTx2_back =
        type_convert<T2>(f4_convert_sr(vTx2, type_convert<float>(scale2))); // expect {0.5, -2}
    p_test[i++] = vTx2_back[0];
    p_test[i++] = vTx2_back[1];
    if(i >= N)
    {
        return;
    }

    /// Test round to nearest even
    p_test[i++] = type_convert<T>(f4_convert_rne(type_convert<T>(24.0f), 4.0f)); // 24/4
    if(i >= N)
    {
        return;
    }

    T v_qnan     = ck::NumericLimits<T>::QuietNaN();
    T v_infinity = ck::NumericLimits<T>::Infinity();
#if !CK_USE_LLVM_BUILTIN_BF16
    if constexpr(std::is_same_v<T, bhalf_t>)
    {
        v_qnan     = bhalf_t{0x7FFF};
        v_infinity = bhalf_t{0x7F80};
    }
#endif
    p_test[i++] = type_convert<T>(f4_convert_rne(v_qnan, 4.0f)); // => NaN
    if(i >= N)
    {
        return;
    }

    // Inf/2 > 6.0 => 6.0 on device
    p_test[i++] = type_convert<T>(f4_convert_rne(v_infinity, 2.0f));
    if(i >= N)
    {
        return;
    }

    // 256/0.5  > 6.0 => 6.0 on device
    p_test[i++] = type_convert<T>(f4_convert_rne(type_convert<T>(256.0f), 0.5f));
    if(i >= N)
    {
        return;
    }

    // -256/0.5  < -6.0 => -6.0 on device
    p_test[i++] = type_convert<T>(f4_convert_rne(type_convert<T>(-256.0f), 0.5f));
    if(i >= N)
    {
        return;
    }

    // proper scale selection
    p_test[i++] = type_convert<T>(f4_convert_rne(type_convert<T>(20.0f), 4.0f)); // 20.0/4.0 = 5.0
    if(i >= N)
    {
        return;
    }
}

template <typename T>
static inline void validate(T* out)
{
    // V = X * P; X - E8M0 scale, P - FP4

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t fp4_id = 0; fp4_id < 16; fp4_id++)
    {
        auto idx = e8m0_nan_id * 16 + fp4_id;
        ASSERT_TRUE(std::isnan(type_convert<float>(out[idx])));
    }

    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t fp4_id = 0; fp4_id < 16; fp4_id++)
        {
            uint8_t fp4_uid = static_cast<uint8_t>(fp4_id);
            auto idx        = exp_id * 16 + fp4_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<T>(type_convert<float>(e8m0_bexp_t(exp_id)) *
                                            type_convert<float>(f4_t(fp4_uid & 0b00001111))))
                << "exp_id: " << exp_id << " fp4_id: " << fp4_id << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(f4_t(fp4_uid & 0b00001111));
        }
    }

    /// Test vector conversions

    auto i = 256 * 16;

    // f4x2 -> f32x2
    EXPECT_EQ(out[i++], type_convert<T>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<T>(1.0f));

    // f32x2 -> f4x2
    // RNE
    EXPECT_EQ(out[i++], type_convert<T>(0.5f));
    EXPECT_EQ(out[i++], type_convert<T>(-2.0f));
    // SR
    EXPECT_EQ(out[i++], type_convert<T>(0.5f));
    EXPECT_EQ(out[i++], type_convert<T>(-2.0f));

    /// Test round to nearest even
    EXPECT_EQ(out[i++], type_convert<T>(24.0f / 4.0f))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(ck::NumericLimits<f4_t>::Lowest()))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);
    EXPECT_EQ(out[i++], type_convert<T>(type_convert<f4_t>(5.0f)))
        << "out[i-1]: " << type_convert<float>(out[i - 1]);

    EXPECT_EQ(test_size, i);
}

TYPED_TEST(MXFP4TypedTest, HostScaledConvert)
{
    using T = TypeParam;
    std::vector<T> out(test_size, T(-1.0f));
    uint64_t completed = 0;

    test_mx_fp4_scaled_convert(test_size, out.data(), &completed);

    EXPECT_EQ(test_size, completed);
    validate(out.data());
}

template <typename T>
__global__ void test_mx_fp4_device_scaled_convert(uint64_t N, T* p_test, uint64_t* p_completed)
{
    test_mx_fp4_scaled_convert(N, p_test, p_completed);
}

TYPED_TEST(MXFP4TypedTest, DeviceScaledConvert)
{
    using T = TypeParam;
    std::vector<T> out(test_size, T(-1.0f));

    DeviceMem device_out(test_size * sizeof(T));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp4_device_scaled_convert<<<1, 1>>>(
        test_size,
        static_cast<T*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    EXPECT_EQ(test_size, completed);
    validate(out.data());
}

__host__ __device__ float vec16_generator(ck::index_t i, float scale)
{
    return scale * type_convert<float>(f4_t(i & 0b00001111));
}

__host__ __device__ float vec32_generator(ck::index_t i, float scale)
{
    if(i < 16)
    {
        return vec16_generator(
            i, scale); // all positive values, then all negative values in ascending order
    }
    else
    {
        return vec16_generator(
            15 - (i % 16),
            scale); // all negative values, then all positive values in descending order
    }
}

template <typename T>
__global__ void test_mx_fp4x32_device_scaled_convert(T* p_test, uint64_t* p_completed)
{
    using T32       = typename ck::vector_type<T, 32>::type;
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

    f4x32_t f4x32{};
    T32 vT32{};
    ck::static_for<0, N, 1>{}([&](auto ii) {
        vT32[static_cast<int>(ii)] =
            type_convert<T>(vec32_generator(ii, type_convert<float>(scale2)));
    });

    f4x32 = f4_convert_rne(vT32, type_convert<float>(scale2));

    ck::static_for<0, N / 2, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<T>(
            f4_t(f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}).template unpack<>(ck::Number<0>{})));
        p_test[i++] = type_convert<T>(
            f4_t(f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}).template unpack<>(ck::Number<1>{})));
    });
}

TYPED_TEST(MXFP4TypedTest, DeviceTx32ToF4x32ScaledConvert)
{
    using T         = TypeParam;
    constexpr int N = 32;
    std::vector<T> out(N, T(-1.0f));

    DeviceMem device_out(N * sizeof(T));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp4x32_device_scaled_convert<<<1, 1>>>(
        static_cast<T*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i      = 0;
    auto scale2 = e8m0_bexp_t(2.0f);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++],
                  type_convert<T>(vec32_generator(ii, type_convert<float>(scale2)) /
                                  type_convert<float>(scale2)))
            << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

template <typename T>
__global__ void test_mx_fp4x32_device_scaled_convert_sr(T* p_test, uint64_t* p_completed)
{
    using T32       = typename ck::vector_type<T, 32>::type;
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

    f4x32_t f4x32{};
    T32 vT32{};
    ck::static_for<0, N, 1>{}([&](auto ii) {
        vT32[static_cast<int>(ii)] =
            type_convert<T>(vec32_generator(ii, type_convert<float>(scale2)));
    });

    f4x32          = f4_convert_sr(vT32, type_convert<float>(scale2));
    auto vT32_back = type_convert<T32>(f4x32);
    ck::static_for<0, N, 1>{}([&](auto ii) { p_test[i++] = vT32_back[static_cast<int>(ii)]; });
}

TYPED_TEST(MXFP4TypedTest, DeviceTx32ToF4x32ScaledConvertSR)
{
    using T         = TypeParam;
    constexpr int N = 32;
    std::vector<T> out(N, T(-1.0f));

    DeviceMem device_out(N * sizeof(T));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp4x32_device_scaled_convert_sr<<<1, 1>>>(
        static_cast<T*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i      = 0;
    auto scale2 = e8m0_bexp_t(2.0f);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++],
                  type_convert<T>(vec32_generator(ii, type_convert<float>(scale2)) /
                                  type_convert<float>(scale2)))
            << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

template <typename T>
__global__ void test_mx_Tx32_device_scaled_convert(T* p_test, uint64_t* p_completed)
{
    using T32       = typename ck::vector_type<T, 32>::type;
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

    f4x32_t f4x32{};
    T32 vT32{};
    ck::static_for<0, N / 2, 1>{}([&](auto ii) {
        f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}) = f4x2_pk_t{}.pack(
            type_convert<f4_t>(vec32_generator(2 * ii, type_convert<float>(scale2)) /
                               type_convert<float>(scale2)),
            type_convert<f4_t>(vec32_generator(2 * ii + 1, type_convert<float>(scale2)) /
                               type_convert<float>(scale2)));
    });

    vT32 = scaled_type_convert<T32>(scale2, f4x32);

    ck::static_for<0, N, 1>{}([&](auto ii) { p_test[i++] = vT32[static_cast<int>(ii)]; });
}

TYPED_TEST(MXFP4TypedTest, DeviceF4x32ToTx32ScaledConvert)
{
    using T         = TypeParam;
    constexpr int N = 32;
    std::vector<T> out(N, T(-1.0f));

    DeviceMem device_out(N * sizeof(T));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_Tx32_device_scaled_convert<<<1, 1>>>(
        static_cast<T*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i      = 0;
    auto scale2 = e8m0_bexp_t(2.0f);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], type_convert<T>(vec32_generator(ii, type_convert<float>(scale2))))
            << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

// Host
TEST(MXFP4, HostF4x32ToF32x32ScaledConvert)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    auto scale2 = e8m0_bexp_t(2.0f);

    f4x32_t f4x32{};
    float32_t vT32{};

    // Fill f4x32 with converted values
    ck::static_for<0, N / 2, 1>{}([&](auto ii) {
        f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}) = f4x2_pk_t{}.pack(
            type_convert<f4_t>(vec32_generator(2 * ii, type_convert<float>(scale2)) /
                               type_convert<float>(scale2)),
            type_convert<f4_t>(vec32_generator(2 * ii + 1, type_convert<float>(scale2)) /
                               type_convert<float>(scale2)));
    });

    // Convert f4x32 to float32 using scaled_type_convert
    vT32 = scaled_type_convert<float32_t>(scale2, f4x32);

    // Extract the values to output vector
    ck::static_for<0, N, 1>{}([&](auto ii) { out[ii] = vT32[static_cast<int>(ii)]; });

    // Verify the output matches expected values
    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[ii], vec32_generator(ii, type_convert<float>(scale2)))
            << "ii: " << ii << std::endl;
    });
}
