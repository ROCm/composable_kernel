// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"
#include "ck/host_utility/device_prop.hpp"

using ck::bhalf_t;
using ck::DeviceMem;
using ck::f6_t;
using ck::f6x16_pk_t;
using ck::f6x16_t;
using ck::half_t;
using ck::type_convert;

template <typename T>
class MXFP6Pk4ScaleTypedTest : public ::testing::Test
{
};

using TestTypes = ::testing::Types<float, half_t, bhalf_t>;

TYPED_TEST_SUITE(MXFP6Pk4ScaleTypedTest, TestTypes);

/* helper function to convert ith scale in packed form to a float */
static inline float convert_exponent_to_float(uint32_t exp4, int i)
{
    return ck::bit_cast<float>((exp4 >> (i * 8) & 0xFF) << 23);
}

/**
 * @brief Device version of "wave-wise FP6 to FP32/FP16/BF16 conversion Block32 mode".
 *
 * This function performs packed 16 conversions from FP6 values to T values in a wave.
 * One packed scale parameter can hold scale factor for 4 conversion calls.
 * See how template parameter Scale_sel used to select scale in the packed form.
 * scale factor [0, 1, 2, 3]
 *
 * @param p_mat Pointer to the output array where the converted T values will be stored.
 * @param p_scale Pointer to the scale array.
 *
 */
template <int M, int N, float Val, typename T>
__global__ void test_packed_scaled_convert_block32(T* p_mat, uint32_t* p_scale)
{
    if(p_mat == nullptr || p_scale == nullptr)
    {
        return;
    }
#if CK_MX_ARCH_125
    using T16       = typename ck::vector_type<T, 16>::type;
    ck::index_t lid = __lane_id();
    uint32_t scale  = p_scale[lid];

    // Each iteration take care of 16 x 32 matrix
    // itr-0, scale_op-0, use scale[th0:15]   [7:0]-th0:15, [15:8]-th16:32
    // itr-1, scale_op-1, use scale[th16:31]  [7:0]-th0:15, [15:8]-th16:32
    // itr-2, scale_op-2, use scale[th0:15]   [23:16]-th0:15, [31:24]-th16:32
    // itr-3, scale_op-3, use scale[th16:31]  [23:16]-th0:15, [31:24]-th16:32
    ck::static_for<0, 4, 1>{}([&](auto it) { // 4 scale factor test
        // 16x32 sub-matrix will be processed by a wave
        // Create two f6x16_t vectors with value Val
        f6x16_pk_t f6x16_pk_v{};
        for(int i = 0; i < 16; i++)
        {
            f6x16_pk_v.pack(type_convert<f6_t>(Val), i);
        }

        f6x16_t vf6{f6x16_pk_v};

        auto vT16 = ck::pk4scaled_type_convert<T16, f6x16_t, it>(scale, vf6);

        /* Row index of p_mat:
         * (lid & 0x0F): mapping lane0-15 and 16-31 to row 0-15
         * Column index of p_mat:
         *  it * 32: each iteration process 32 columns
         * ((lid >> 4) & 1) * 16: lane0-15 write first 16 column
         *                        lane 16-31 write the next 16 columns*/
        ck::static_for<0, 16, 1>{}([&](auto ii) {
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii] =
                vT16[static_cast<int>(ii)];
        });
    });
#endif
}

TYPED_TEST(MXFP6Pk4ScaleTypedTest, DeviceWavewiseBlock32)
{
    using T = TypeParam;

    // matrix shape M x N
    constexpr int M     = 16;
    constexpr int N     = 128;
    constexpr float Val = 2.0f;
    std::vector<T> out(M * N, -1.0f);
    std::vector<uint32_t> scale(2 * M);
    // Test scale variations: 16 rows × 128 columns
    // - Each row has different 8 scale factor (scale[m]-packed 4 and scale[m+16]-packed4)
    // - Within a row, every 16 consecutive columns share the same scale factor
    for(int m = 0; m < M; m++)
    {
        // scale[m]: threads 0-15
        scale[m] = ((124u + (m % 6)) << 24) | // Byte3: cycles 0.125, 0.25, 0.5, 1, 2, 4
                   ((125u + (m % 3)) << 16) | // Byte2: cycles 0.25, 0.5, 1
                   ((126u + (m % 4)) << 8) |  // Byte1: cycles 0.5, 1, 2, 4
                   (127u + (m % 5));          // Byte0: cycles 1, 2, 4, 8, 16

        // scale[m+M]: threads 16-31
        scale[m + M] = ((131u - (m % 6)) << 24) | // Byte3: cycles 16, 8, 4, 2, 1, 0.5
                       ((126u + (m % 5)) << 16) | // Byte2: cycles 0.5, 1, 2, 4, 8
                       ((128u + (m % 3)) << 8) |  // Byte1: cycles 2, 4, 8
                       (130u - (m % 4));          // Byte0: cycles 8, 4, 2, 1
    }

    DeviceMem device_out(M * N * sizeof(T));
    DeviceMem device_scale(2 * M * sizeof(uint32_t));
    device_scale.ToDevice(scale.data());

    test_packed_scaled_convert_block32<M, N, Val>
        <<<1, 32>>>(static_cast<T*>(device_out.GetDeviceBuffer()),
                    static_cast<uint32_t*>(device_scale.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* every 16 columns in a row share one scale factor */
    for(int m = 0; m < M; m++)
    {
        /* n = [0:31] */
        for(int n = 0; n < 16; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m], 1) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [32:63] */
        for(int n = 32; n < 48; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 1) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [64:95] */
        for(int n = 64; n < 80; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m], 3) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [96:127] */
        for(int n = 96; n < 112; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 3) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
    }
}

/**
 * @brief Device version of "wave-wise FP6 to FP32/FP16/BF16 conversion Block16 mode".
 *
 * This function performs packed 16 conversions from FP6 values to T values in a wave.
 * One packed scale parameter can hold scale factor for 4 conversion calls.
 * See how template parameter Scale_sel used to select scale in the packed form.
 * scale factor [4, 5, 6, 7]
 *
 * @param p_mat Pointer to the output array where the converted T values will be stored.
 * @param p_scale Pointer to the scale array.
 *
 */
template <int M, int N, float Val, typename T>
__global__ void test_packed_scaled_convert_block16(T* p_mat, uint32_t* p_scale)
{
    if(p_mat == nullptr || p_scale == nullptr)
    {
        return;
    }
#if CK_MX_ARCH_125
    using T16       = typename ck::vector_type<T, 16>::type;
    ck::index_t lid = __lane_id();
    uint32_t scale  = p_scale[lid];

    // Each iteration take care of 16 x 32 matrix
    // itr-0, use scale[th0:15]   [7:0]-th0:15, [[23:16]-th16:32
    // itr-1, use scale[th16:31]  [7:0]-th0:15, [23:16]-th16:32
    // itr-2, use scale[th0:15]   [15:8]-th0:15, [31:24]-th16:32
    // itr-3, use scale[th16:31]  [15:8]-th0:15, [31:24]-th16:32
    ck::static_for<0, 4, 1>{}([&](auto it) { // 4 scale factor test
        // 16x32 sub-matrix will be processed by a wave
        // Create two f6x16_t vectors with value Val
        f6x16_pk_t f6x16_pk_v{};
        for(int i = 0; i < 16; i++)
        {
            f6x16_pk_v.pack(type_convert<f6_t>(Val), i);
        }

        f6x16_t vf6{f6x16_pk_v};

        auto vT16 = ck::pk4scaled_type_convert<T16, f6x16_t, it + 4>(scale, vf6);

        /* Row index of p_mat:
         * (lid & 0x0F): mapping lane0-15 and 16-31 to row 0-15
         * Column index of p_mat:
         *  it * 32: each iteration process 32 columns
         * ((lid >> 4) & 1) * 16: lane0-15 write first 16 column
         *                        lane 16-31 write the next 16 columns*/
        ck::static_for<0, 16, 1>{}([&](auto ii) {
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii] =
                vT16[static_cast<int>(ii)];
        });
    });
#endif
}

TYPED_TEST(MXFP6Pk4ScaleTypedTest, DeviceWavewiseBlock16)
{
    if(ck::is_gfx125_supported() && ck::get_device_revision() == 0)
    {
        // Block16 Mode here means scale option [4-7].
        GTEST_SKIP() << "Block16 Mode not supported on asicRevision=0";
    }
    using T = TypeParam;

    // matrix shape M x N
    constexpr int M     = 16;
    constexpr int N     = 128;
    constexpr float Val = 2.0f;
    std::vector<T> out(M * N, -1.0f);
    std::vector<uint32_t> scale(2 * M);
    // Test scale variations: 16 rows × 128 columns
    // - Each row has different 8 scale factor (scale[m]-packed 4 and scale[m+16]-packed4)
    // - Within a row, every 16 consecutive columns share the same scale factor
    for(int m = 0; m < M; m++)
    {
        // scale[m]: threads 0-15
        scale[m] = ((124u + (m % 6)) << 24) | // Byte3: cycles 0.125, 0.25, 0.5, 1, 2, 4
                   ((125u + (m % 3)) << 16) | // Byte2: cycles 0.25, 0.5, 1
                   ((126u + (m % 4)) << 8) |  // Byte1: cycles 0.5, 1, 2, 4
                   (127u + (m % 5));          // Byte0: cycles 1, 2, 4, 8, 16

        // scale[m+M]: threads 16-31
        scale[m + M] = ((131u - (m % 6)) << 24) | // Byte3: cycles 16, 8, 4, 2, 1, 0.5
                       ((126u + (m % 5)) << 16) | // Byte2: cycles 0.5, 1, 2, 4, 8
                       ((128u + (m % 3)) << 8) |  // Byte1: cycles 2, 4, 8
                       (130u - (m % 4));          // Byte0: cycles 8, 4, 2, 1
    }

    DeviceMem device_out(M * N * sizeof(T));
    DeviceMem device_scale(2 * M * sizeof(uint32_t));
    device_scale.ToDevice(scale.data());

    test_packed_scaled_convert_block16<M, N, Val>
        <<<1, 32>>>(static_cast<T*>(device_out.GetDeviceBuffer()),
                    static_cast<uint32_t*>(device_scale.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* every 16 columns in a row share one scale factor */
    for(int m = 0; m < M; m++)
    {
        /* n = [0:31] */
        for(int n = 0; n < 16; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m], 2) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [32:63] */
        for(int n = 32; n < 48; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 2) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [64:95] */
        for(int n = 64; n < 80; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 1) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m], 3) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [96:127] */
        for(int n = 96; n < 112; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 1) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 3) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
    }
}
