// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <cmath>
#include <vector>

#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using ck_tile::bf16_to_float;
using ck_tile::bf16x2_t;
using ck_tile::bfloat16_t;
using ck_tile::bit_cast;
using ck_tile::float_to_bf16;
using ck_tile::fp32x2_t;

// =====================================================================
// Tests for bf16x2_to_fp32x2 (host-side, always available)
// =====================================================================

TEST(Bf16F32Convert, Bf16x2ToFp32x2_BasicValues)
{
    auto a = float_to_bf16(1.0f);
    auto b = float_to_bf16(-2.5f);

    bf16x2_t packed{a, b};
    fp32x2_t result = ck_tile::bf16x2_to_fp32x2(packed);

    EXPECT_FLOAT_EQ(result[0], bf16_to_float(a));
    EXPECT_FLOAT_EQ(result[1], bf16_to_float(b));
}

TEST(Bf16F32Convert, Bf16x2ToFp32x2_Zeros)
{
    auto pos_zero = float_to_bf16(0.0f);
    auto neg_zero = float_to_bf16(-0.0f);

    bf16x2_t packed{pos_zero, neg_zero};
    fp32x2_t result = ck_tile::bf16x2_to_fp32x2(packed);

    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_TRUE(std::signbit(result[1]));
    EXPECT_FLOAT_EQ(result[1], -0.0f);
}

TEST(Bf16F32Convert, Bf16x2ToFp32x2_LargeSmall)
{
    auto big   = float_to_bf16(65504.0f);
    auto small = float_to_bf16(0.00390625f);

    bf16x2_t packed{big, small};
    fp32x2_t result = ck_tile::bf16x2_to_fp32x2(packed);

    EXPECT_FLOAT_EQ(result[0], bf16_to_float(big));
    EXPECT_FLOAT_EQ(result[1], bf16_to_float(small));
}

TEST(Bf16F32Convert, Bf16x2ToFp32x2_RoundTrip)
{
    const float test_values[] = {1.0f, -1.0f, 0.5f, 3.14f, 100.0f, -42.0f, 0.001f};
    for(float v : test_values)
    {
        auto bf        = float_to_bf16(v);
        float expected = bf16_to_float(bf);

        bf16x2_t packed{bf, bf};
        fp32x2_t result = ck_tile::bf16x2_to_fp32x2(packed);

        EXPECT_FLOAT_EQ(result[0], expected) << "v=" << v;
        EXPECT_FLOAT_EQ(result[1], expected) << "v=" << v;
    }
}

// =====================================================================
// Tests for fp32x2_to_bf16x2 (host-side)
// =====================================================================

TEST(Bf16F32Convert, Fp32x2ToBf16x2_BasicValues)
{
    fp32x2_t input{1.5f, -3.0f};
    bf16x2_t result = ck_tile::fp32x2_to_bf16x2(input);

    EXPECT_FLOAT_EQ(bf16_to_float(result[0]), bf16_to_float(float_to_bf16(1.5f)));
    EXPECT_FLOAT_EQ(bf16_to_float(result[1]), bf16_to_float(float_to_bf16(-3.0f)));
}

// =====================================================================
// Device tests for cvt_pk_bf16_f32 and convert_float_to_bf16_pairs
// =====================================================================

struct CvtPkBf16F32Result
{
    bfloat16_t r0;
    bfloat16_t r1;
};

__global__ void kernel_cvt_pk_bf16_f32(const float* in, CvtPkBf16F32Result* out, int n)
{
    int idx = threadIdx.x;
    if(idx < n)
    {
        bf16x2_t result = ck_tile::cvt_pk_bf16_f32(in[2 * idx], in[2 * idx + 1]);
        out[idx].r0     = result[0];
        out[idx].r1     = result[1];
    }
}

TEST(Bf16F32Convert, CvtPkBf16F32_Device)
{
    const std::vector<float> host_in = {1.0f, -1.0f, 0.0f, 3.14f, 100.0f, -0.5f, 42.0f, 0.001f};
    const int num_pairs              = host_in.size() / 2;

    ck_tile::DeviceMem in_buf(host_in.size() * sizeof(float));
    ck_tile::DeviceMem out_buf(num_pairs * sizeof(CvtPkBf16F32Result));
    in_buf.ToDevice(host_in.data());

    kernel_cvt_pk_bf16_f32<<<1, num_pairs>>>(
        static_cast<const float*>(in_buf.GetDeviceBuffer()),
        static_cast<CvtPkBf16F32Result*>(out_buf.GetDeviceBuffer()),
        num_pairs);
    (void)hipDeviceSynchronize();

    std::vector<CvtPkBf16F32Result> host_out(num_pairs);
    out_buf.FromDevice(host_out.data());

    for(int i = 0; i < num_pairs; i++)
    {
        float ref0 = bf16_to_float(float_to_bf16(host_in[2 * i]));
        float ref1 = bf16_to_float(float_to_bf16(host_in[2 * i + 1]));
        EXPECT_FLOAT_EQ(bf16_to_float(host_out[i].r0), ref0) << "pair=" << i << " elem=0";
        EXPECT_FLOAT_EQ(bf16_to_float(host_out[i].r1), ref1) << "pair=" << i << " elem=1";
    }
}

// =====================================================================
// Device test for convert_float_to_bf16_pairs
// =====================================================================

template <int VecSize>
struct Bf16PairsResult
{
    bfloat16_t big[VecSize];
    bfloat16_t small_val[VecSize];
};

template <int VecSize>
__global__ void kernel_convert_float_to_bf16_pairs(const float* in, Bf16PairsResult<VecSize>* out)
{
    using float_vec_t = ck_tile::ext_vector_t<float, VecSize>;
    using bf16_vec_t  = ck_tile::ext_vector_t<bfloat16_t, VecSize>;

    float_vec_t reg_f32;
    for(int i = 0; i < VecSize; i++)
        reg_f32[i] = in[i];

    bf16_vec_t reg_big, reg_small;
    ck_tile::convert_float_to_bf16_pairs<VecSize>(reg_f32, reg_big, reg_small);

    for(int i = 0; i < VecSize; i++)
    {
        out[0].big[i]       = reg_big[i];
        out[0].small_val[i] = reg_small[i];
    }
}

template <int VecSize>
void test_convert_float_to_bf16_pairs_device()
{
    static_assert(VecSize >= 2 && VecSize % 2 == 0);

    std::vector<float> host_in(VecSize);
    // Use diverse values: mix of exact and non-exact bf16 representable numbers
    const float base_vals[] = {1.1f, -2.3f, 0.7f, 100.1f, -0.001f, 42.42f, 3.14f, -7.77f};
    for(int i = 0; i < VecSize; i++)
        host_in[i] = base_vals[i % 8];

    ck_tile::DeviceMem in_buf(VecSize * sizeof(float));
    ck_tile::DeviceMem out_buf(sizeof(Bf16PairsResult<VecSize>));
    in_buf.ToDevice(host_in.data());

    kernel_convert_float_to_bf16_pairs<VecSize>
        <<<1, 1>>>(static_cast<const float*>(in_buf.GetDeviceBuffer()),
                   static_cast<Bf16PairsResult<VecSize>*>(out_buf.GetDeviceBuffer()));
    (void)hipDeviceSynchronize();

    Bf16PairsResult<VecSize> host_out;
    out_buf.FromDevice(&host_out);

    for(int i = 0; i < VecSize; i++)
    {
        float orig  = host_in[i];
        float big_f = bf16_to_float(host_out.big[i]);

        // big should match scalar float_to_bf16
        float ref_big = bf16_to_float(float_to_bf16(orig));
        EXPECT_FLOAT_EQ(big_f, ref_big) << "VecSize=" << VecSize << " i=" << i;

        // small should match float_to_bf16(orig - big)
        float ref_small = bf16_to_float(float_to_bf16(orig - ref_big));
        float small_f   = bf16_to_float(host_out.small_val[i]);
        EXPECT_FLOAT_EQ(small_f, ref_small) << "VecSize=" << VecSize << " i=" << i;

        // big + small should be closer to orig than big alone
        float reconstructed = big_f + small_f;
        EXPECT_LE(std::fabs(reconstructed - orig), std::fabs(big_f - orig) + 1e-10f)
            << "VecSize=" << VecSize << " i=" << i;
    }
}

TEST(Bf16F32Convert, ConvertFloatToBf16Pairs_Vec2) { test_convert_float_to_bf16_pairs_device<2>(); }
TEST(Bf16F32Convert, ConvertFloatToBf16Pairs_Vec4) { test_convert_float_to_bf16_pairs_device<4>(); }
TEST(Bf16F32Convert, ConvertFloatToBf16Pairs_Vec8) { test_convert_float_to_bf16_pairs_device<8>(); }

// =====================================================================
// 3x BF16 multiply-accumulate precision test
// =====================================================================

TEST(Bf16F32Convert, ThreeBf16MulAccPrecision)
{
    // Verify that a_big*b_big + a_small*b_big + a_big*b_small is more precise
    // than a single bf16(a)*bf16(b) for non-exact values
    const float test_pairs[][2] = {
        {1.1f, 2.3f}, {3.14f, -2.71f}, {0.123f, 456.789f}, {-100.1f, 0.99f}};

    for(const auto& pair : test_pairs)
    {
        float a = pair[0];
        float b = pair[1];

        float a_big_f   = bf16_to_float(float_to_bf16(a));
        float a_small_f = bf16_to_float(float_to_bf16(a - a_big_f));
        float b_big_f   = bf16_to_float(float_to_bf16(b));
        float b_small_f = bf16_to_float(float_to_bf16(b - b_big_f));

        float exact       = a * b;
        float single_bf16 = a_big_f * b_big_f;
        float three_bf16  = a_big_f * b_big_f + a_small_f * b_big_f + a_big_f * b_small_f;

        float err_single = std::fabs(exact - single_bf16);
        float err_three  = std::fabs(exact - three_bf16);

        EXPECT_LE(err_three, err_single + 1e-10f)
            << "a=" << a << " b=" << b << " exact=" << exact << " single=" << single_bf16
            << " three=" << three_bf16;
    }
}
