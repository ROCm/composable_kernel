// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"
#include "ck/host_utility/device_prop.hpp"

using ck::bf8_ocp_t;
using ck::bf8x8_ocp_t;
using ck::bhalf_t;
using ::ck::DeviceMem;
using ck::half_t;
using ck::type_convert;

template <typename T>
class MXBF8TypedTest : public ::testing::Test
{
};

using TestTypes = ::testing::Types<float, half_t, bhalf_t>;
TYPED_TEST_SUITE(MXBF8TypedTest, TestTypes);

/* helper function to convert ith scale in packed form to a float */
static inline float convert_exponent_to_float(uint32_t exp4, int i)
{
    return ck::bit_cast<float>((exp4 >> (i * 8) & 0xFF) << 23);
}

/**
 * @brief Device version of "wave-wise BF8 to T conversion".
 *
 * This function performs packed 8 conversions from BF8 values to T values in a wave.
 * One packed scale parameter can hold scale factor for 4 conversion calls.
 * See how template parameter Scale_sel used to select scale in the packed form.
 *
 * @param p_mat Pointer to the output array where the converted T values will be stored.
 * @param p_scale Pointer to the scale array.
 *
 */
template <int M, int N, float Val, typename T>
__global__ void test_packed_scaled_convert(T* p_mat, uint32_t* p_scale)
{
    if(p_mat == nullptr || p_scale == nullptr)
    {
        return;
    }

#if CK_MX_ARCH_125
    using T8        = typename ck::vector_type<T, 8>::type;
    ck::index_t lid = __lane_id();
    // scale_sel = 1, 3, 5, 6 will use p_scale values in lane[16:31]
    uint32_t scale = (lid < 16) ? uint32_t(0) : p_scale[lid - 16];

    // Each iteration take care of 16 x 32 matrix
    /* itr-0, scale_sel = 1 : src * scale[th[16:31]][7:0]
     * itr-1, scale_sel = 3 : src * scale[th[16:31]][23:16]
     * itr-2, scale_sel = 5 : src * scale[th[16:31]][15:8]]
     * itr-3, scale_sel = 7 : src * scale[th[16:31]][31:24]*/
    ck::static_for<0, 4, 1>{}([&](auto it) { // 4 scale factor test
        // 16x32 sub-matrix will be processed by a wave
        bf8x8_ocp_t vf8_1{type_convert<bf8_ocp_t>(Val)}; // 2.0f
        bf8x8_ocp_t vf8_2{type_convert<bf8_ocp_t>(Val)}; // 2.0f
        auto vT8_1 = ck::pk4scaled_type_convert<T8, bf8x8_ocp_t, it * 2 + 1>(scale, vf8_1);
        auto vT8_2 = ck::pk4scaled_type_convert<T8, bf8x8_ocp_t, it * 2 + 1>(scale, vf8_2);

        // write to p_mat
        ck::static_for<0, 8, 1>{}([&](auto ii) {
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii] =
                vT8_1[static_cast<int>(ii)];
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii + 8] =
                vT8_2[static_cast<int>(ii)];
        });
    });
#endif
}

TYPED_TEST(MXBF8TypedTest, DeviceWavewiseBlock32)
{
    using T = TypeParam;
    // matrix shape M x N
    constexpr int M     = 16;
    constexpr int N     = 128; // Block 32 share a scale factor, 4 scale factors available
    constexpr float Val = 2.0f;
    uint32_t v_scal     = (126u << 24) | (127u << 16) | (128u << 8) | (129u); //[0.5|1.|2.|4.]
    std::vector<T> out(M * N, -1.0f);
    std::vector<uint32_t> scale(M, v_scal);

    DeviceMem device_out(M * N * sizeof(T));
    DeviceMem device_scale(M * sizeof(uint32_t));
    device_scale.ToDevice(scale.data());

    test_packed_scaled_convert<M, N, Val>
        <<<1, 32>>>(static_cast<T*>(device_out.GetDeviceBuffer()),
                    static_cast<uint32_t*>(device_scale.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* n:  [0:31]  [32:63]  [64:95]  [96:127]
            8.0f     2.0f     4.0f    1.0f  */
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < 32; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 32; n < 64; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 64; n < 96; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 1) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 96; n < 128; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 3) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
    }
}

// block 16
/**
 * @brief Device version of "wave-wise BF8 to FP32/F16/BF16 block16 conversion".
 *
 * This function performs packed 8 conversions from BF8 values to T values in a wave.
 * One packed scale parameter can hold scale factor for 4 conversion calls.
 * See how template parameter Scale_sel used to select scale in the packed form.
 *
 * @param p_mat Pointer to the output array where the converted T values will be stored.
 * @param p_scale Pointer to the scale array.
 *
 */
template <int M, int N, float Val, typename T>
__global__ void test_packed_scaled_block16_convert(T* p_mat, uint32_t* p_scale)
{
    if(p_mat == nullptr || p_scale == nullptr)
    {
        return;
    }
#if CK_MX_ARCH_125
    using T8        = typename ck::vector_type<T, 8>::type;
    ck::index_t lid = __lane_id();
    uint32_t scale  = p_scale[lid];

    // Each iteration take care of 16 x 32 matrix
    /* itr-0, scale_sel = 8 : src[th[0:15]]  * scale[th[0:15]][7:0]
     *                        src[th[16:31]] * scale[th[0:15]][15:8]
     * itr-1, scale_sel = 9 : src[th[0:15]]  * scale[th[16:31]][7:0]
     *                        src[th[16:31]] * scale[th[16:31]][15:8]
     * itr-2, scale_sel = 10: src[th[0:15]]  * scale[th[0:15]][23:16]
     *                        src[th[16:31]] * scale[th[0:15]][31:24]
     * itr-3, scale_sel = 11: src[th[0:15]]  * scale[th[16:31]][23:16]
     *                       src[th[16:31]] * scale[th[16:31]][31:24] */
    ck::static_for<0, 4, 1>{}([&](auto it) { // 4 scale factor test
        // 16x32 sub-matrix will be processed by a wave
        bf8x8_ocp_t vf8_1{type_convert<bf8_ocp_t>(Val)}; // 2.0f
        bf8x8_ocp_t vf8_2{type_convert<bf8_ocp_t>(Val)}; // 2.0f
        auto vT8_1 = ck::pk4scaled_type_convert<T8, bf8x8_ocp_t, it + 8>(scale, vf8_1);
        auto vT8_2 = ck::pk4scaled_type_convert<T8, bf8x8_ocp_t, it + 8>(scale, vf8_2);

        // write to p_mat
        ck::static_for<0, 8, 1>{}([&](auto ii) {
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii] =
                vT8_1[static_cast<int>(ii)];
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii + 8] =
                vT8_2[static_cast<int>(ii)];
        });
    });
#endif
}

TYPED_TEST(MXBF8TypedTest, DeviceWavewiseBlock16)
{
    if(ck::is_gfx125_supported() && ck::get_device_revision() == 0)
    {
        // Block16 Mode here means scale option [8-11].
        GTEST_SKIP() << "Block16 Mode not supported on asicRevision=0";
    }

    using T = TypeParam;
    // matrix shape M x N
    constexpr int M     = 16;
    constexpr int N     = 128; // Block 16 share a scale factor, 4 scale factors available
    constexpr float Val = 2.0f;
    std::vector<T> out(M * N, -1.0f);
    std::vector<uint32_t> scale(M * 2, 0);

    for(int i = 0; i < M; i++)
    {
        scale[i]     = (126u << 24) | (127u << 16) | (128u << 8) | (129u); //[0.5|1.|2.|4.]
        scale[M + i] = (135u << 24) | (123u << 16) | (133u << 8) | (120u); //[2^8|2^-4|2^6|2^-7]
    }

    DeviceMem device_out(M * N * sizeof(T));
    DeviceMem device_scale(M * 2 * sizeof(uint32_t));
    device_scale.ToDevice(scale.data());

    test_packed_scaled_block16_convert<M, N, Val>
        <<<1, 32>>>(static_cast<T*>(device_out.GetDeviceBuffer()),
                    static_cast<uint32_t*>(device_scale.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* n:  [0:15]  [16:31]  [32:47]  [48:63]  [64:79]   [80:95]  [96:111]   [112:127]
            8.0f     4.0f     2^-6    2^7      2.0f       1.0f     2^-3       2^9  */
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < 16; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;

            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m], 1) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        for(int n = 32; n < 48; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<T>(convert_exponent_to_float(scale[M + m], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[M + m], 1) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        for(int n = 64; n < 80; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m], 3) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        for(int n = 96; n < 112; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<T>(convert_exponent_to_float(scale[M + m], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[M + m], 3) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
    }
}
