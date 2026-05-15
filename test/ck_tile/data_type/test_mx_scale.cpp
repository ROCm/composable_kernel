// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <hip/hip_runtime.h>
#include <cmath>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using ck_tile::bf16_t;
using ck_tile::bf16x2_t;
using ck_tile::bf16x8_t;
using ck_tile::bf8_t;
using ck_tile::bf8x2_t;
using ck_tile::fp16_t;
using ck_tile::fp16x2_t;
using ck_tile::fp16x8_t;
using ck_tile::fp32_t;
using ck_tile::fp32x2_t;
using ck_tile::fp32x8_t;
using ck_tile::fp8_t;
using ck_tile::fp8x2_t;
using ck_tile::number;
using ck_tile::pk_fp4_t;

using ck_tile::Packed4Scale_E8M0;
using ck_tile::scaled_type_convert;
using ck_tile::type_convert;

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_convert();

template <typename TF8, typename T, bool is_device>
CK_TILE_HOST void test_f8scaled_convert();

template <typename Src, typename DST, bool Block16Mod = false>
void test_f8_pkscale_type_convert_device();

using ck_tile::e8m0_raw_t;
using ck_tile::e8m0_t;

TEST(OCP_Scale, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<e8m0_t>::has_inf(), false);
    EXPECT_EQ(ck_tile::numeric<e8m0_t>::zero(), ck_tile::numeric<e8m0_t>::signaling_NaN());
    EXPECT_EQ(ck_tile::numeric<e8m0_t>::min(), e8m0_t{e8m0_raw_t{0b00000000}});
    EXPECT_EQ(ck_tile::numeric<e8m0_t>::max(), e8m0_t{e8m0_raw_t{0b11111110}});
}
TEST(OCP_Scale, NumericBasic)
{
    auto scale_1 = e8m0_t{1.0f};
    auto scale_2 = e8m0_t{e8m0_raw_t{ck_tile::numeric_traits<e8m0_t>::bias}}; // 2^0
    EXPECT_EQ(scale_1, scale_2);

    auto scale_3 = e8m0_t{8.0f};
    auto scale_4 = e8m0_t{e8m0_raw_t{3 + ck_tile::numeric_traits<e8m0_t>::bias}}; // 2^3
    EXPECT_EQ(scale_3, scale_4);
}

TEST(OCP_Scale, ScaledConvertDevice)
{
    constexpr bool is_device = true;
    test_convert<fp32_t, fp32_t, is_device>(); // fp32 -> fp4 -> fp32
    test_convert<fp16_t, fp16_t, is_device>();
    test_convert<bf16_t, bf16_t, is_device>();
    test_convert<fp32_t, fp16_t, is_device>();
    test_convert<fp32_t, bf16_t, is_device>();
    test_convert<fp16_t, fp32_t, is_device>();
    test_convert<bf16_t, fp32_t, is_device>();
}
TEST(OCP_Scale, ScaledConvertHost)
{
    constexpr bool is_device = false;
    test_convert<fp32_t, fp32_t, is_device>(); // fp32 -> fp4 -> fp32
    test_convert<fp16_t, fp16_t, is_device>();
    test_convert<bf16_t, bf16_t, is_device>();
    test_convert<fp32_t, fp16_t, is_device>();
    test_convert<fp32_t, bf16_t, is_device>();
    test_convert<fp16_t, fp32_t, is_device>();
    test_convert<bf16_t, fp32_t, is_device>();
}
TEST(OCP_Scale, tensorInit)
{
    using scale_t = e8m0_t;
    ck_tile::HostTensor<scale_t> scales({10, 10});
    ck_tile::FillUniformDistribution<scale_t>{1.f, 1.f}(scales);
    scales.SetZero();
}

// Typed test fixture for fp8/bf8 scaled conversion tests
template <typename T>
class F8_OCP_Scale : public ::testing::Test
{
};

using TestTypes = ::testing::Types<float, fp16_t, bf16_t>;

TYPED_TEST_SUITE(F8_OCP_Scale, TestTypes);
TYPED_TEST(F8_OCP_Scale, FP8_Packed8ScaledConvertDevice)
{
    constexpr bool is_device = true;
    test_f8scaled_convert<fp8_t, TypeParam, is_device>();
}

TYPED_TEST(F8_OCP_Scale, FP8_Packed8ScaledConvertHost)
{
    constexpr bool is_device = false;
    test_f8scaled_convert<fp8_t, TypeParam, is_device>();
}

TYPED_TEST(F8_OCP_Scale, BF8_Packed8ScaledConvertDevice)
{
    constexpr bool is_device = true;
    test_f8scaled_convert<bf8_t, TypeParam, is_device>();
}

TYPED_TEST(F8_OCP_Scale, BF8_Packed8ScaledConvertHost)
{
    constexpr bool is_device = false;
    test_f8scaled_convert<bf8_t, TypeParam, is_device>();
}

TYPED_TEST(F8_OCP_Scale, FP8_PkscaleTypeConvertOpsel0_7)
{
    using DstT = TypeParam;

    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }
    test_f8_pkscale_type_convert_device<fp8_t, DstT>();
}

TYPED_TEST(F8_OCP_Scale, BF8_PkscaleTypeConvertOpsel0_7)
{
    using DstT = TypeParam;

    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }
    test_f8_pkscale_type_convert_device<bf8_t, DstT>();
}

TYPED_TEST(F8_OCP_Scale, FP8_PkscaleTypeConvertOpsel8_11)
{
    using DstT = TypeParam;
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }

    if(ck_tile::get_device_revision() == 0)
    {
        // Block16 Mode here means scale option [8-11].
        GTEST_SKIP() << "Block16 Mode not supported on asicRevision=0";
    }
    test_f8_pkscale_type_convert_device<fp8_t, DstT, true>();
}

TYPED_TEST(F8_OCP_Scale, BF8_PkscaleTypeConvertOpsel8_11)
{
    using DstT = TypeParam;
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }

    if(ck_tile::get_device_revision() == 0)
    {
        // Block16 Mode here means scale option [8-11].
        GTEST_SKIP() << "Block16 Mode not supported on asicRevision=0";
    }
    test_f8_pkscale_type_convert_device<bf8_t, DstT, true>();
}

#define toPF4(x, y) ck_tile::scaled_type_convert<pk_fp4_t>(x, y)
#define toDST(x, y) ck_tile::scaled_type_convert<DST>(x, y)
#define toDSTx2(x, y) ck_tile::scaled_type_convert<DSTx2_t>(x, y)

#define toF32(x) ck_tile::type_convert<float>(x)
#define toPF4_(x) ck_tile::type_convert<pk_fp4_t>(x)
#define toSRC(x) ck_tile::type_convert<SRC>(x)
#define toDST_(x) ck_tile::type_convert<DST>(x)

template <typename Kernel, typename... Args>
__global__ void MyKernel(Args... args)
{
    Kernel{}(args...);
}
template <typename SRC, typename DST, int N>
struct SrcPkfp4Dst
{
    CK_TILE_HOST_DEVICE void
    operator()(const SRC* src, DST* dst, e8m0_t scale1, e8m0_t scale2) const
    {

        using SRCx2_t = ck_tile::ext_vector_t<SRC, 2>;
        using DSTx2_t = ck_tile::ext_vector_t<DST, 2>;

        ck_tile::static_for<0, N, 2>{}([&](auto i) {
            const auto input2 = SRCx2_t{src[i], src[i + 1]};

            if(i % 4 == 0)
            {
                // ex: fp32_t -> fp4 -> bf16_t
                dst[i] = toDST(toPF4(src[i], scale1), scale2);
                // ex: fp32x2_t -> pk_fp4 -> unpack<0> -> bf16_t
                dst[i + 1] = toDST(toPF4_(toPF4(input2, scale1).unpack(number<1>{})), scale2);
            }
            else
            {
                // ex: fp32x2_t -> pk_fp4_t -> bf16x2_t
                reinterpret_cast<DSTx2_t*>(dst)[i >> 1] = toDSTx2(toPF4(input2, scale1), scale2);
            }
        });
    }
};

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_convert()
{
    const auto test_data = std::array{4.f, 6.f, 8.f, 10.f};
    const auto ref_data  = std::array{8.f, 16.f, 16.f, 16.f};
    const auto scale1    = e8m0_t{8.0f};
    const auto scale2    = e8m0_t{16.0f};

    static_assert(test_data.size() == ref_data.size());
    static_assert(test_data.size() % 2 == 0);

    constexpr int N = test_data.size();
    std::array<SRC, N> in;
    std::array<DST, N> ref, out;

    // prepare input and ground truth in host
    for(int i = 0; i < N; ++i)
    {
        in[i]  = toSRC(test_data[i]);
        ref[i] = toDST_(ref_data[i]);
        EXPECT_EQ(test_data[i], toF32(in[i]));
        EXPECT_EQ(ref_data[i], toF32(ref[i]));
    }

    using job = SrcPkfp4Dst<SRC, DST, N>;

    if constexpr(is_device)
    {
        auto in_d  = std::make_unique<ck_tile::DeviceMem>(in.size() * sizeof(SRC));
        auto out_d = std::make_unique<ck_tile::DeviceMem>(out.size() * sizeof(DST));
        in_d->ToDevice(in.data());

        MyKernel<job><<<1, 1>>>(reinterpret_cast<const SRC*>(in_d->GetDeviceBuffer()),
                                reinterpret_cast<DST*>(out_d->GetDeviceBuffer()),
                                scale1,
                                scale2);

        out_d->FromDevice(out.data());
    }
    else
    {
        job{}(in.data(), out.data(), scale1, scale2);
    }

    for(int i = 0; i < N; ++i)
        EXPECT_EQ(ref[i], out[i]) << "i:" << i;
}

/* Kernel : test T -> TF8 -> T conversion */
template <typename TF8, typename T, int N>
struct SrcPk8F8Dst
{
    CK_TILE_HOST_DEVICE void
    operator()(const T* in_arr, T* out_arr, const float* scale_div, const float* scale_mul) const
    {
        using TF8x8_t = ck_tile::ext_vector_t<TF8, 8>;
        using Tx8_t   = ck_tile::ext_vector_t<T, 8>;

        const int Nset = N / 8;
        ck_tile::static_for<0, Nset, 1>{}([&](auto i) {
            Tx8_t input8{};
            for(int j = 0; j < 8; ++j)
            {
                input8[j] = in_arr[i * 8 + j];
            }

            auto fp8_packed = ck_tile::scaled_type_convert<TF8x8_t>(input8, scale_div[i]);
            auto output8    = ck_tile::scaled_type_convert<Tx8_t>(fp8_packed, scale_mul[i]);

            for(int j = 0; j < 8; ++j)
                out_arr[i * 8 + j] = output8[j];
        });
    }
};

template <typename TF8, typename T, bool is_device>
CK_TILE_HOST void test_f8scaled_convert()
{
    const auto scale_div = std::array{4.0f, 2.0f};
    const auto scale_mul = std::array{2.0f, 128.0f};

    float maxT8   = toF32(ck_tile::numeric<TF8>::max());
    float maxT    = toF32(ck_tile::numeric<T>::max());
    float lowestT = toF32(ck_tile::numeric<T>::lowest());
    float qnanT   = ck_tile::numeric<float>::quiet_NaN();
    float infT    = ck_tile::numeric<float>::infinity();
    float underflowF8, subNormF8, minNormF8;
    if constexpr(std::is_same_v<TF8, fp8_t>)
    {
        underflowF8 = powf(2.0f, -10.0f);
        subNormF8   = powf(2.0f, -9.0f);
        minNormF8   = powf(2.0f, -6.0f);
    }
    else
    {
        // bf8_t (E5M2): bias=15, min_subnorm=2^-17, min_norm=2^-14
        underflowF8 = powf(2.0f, -18.0f); // below min subnormal
        subNormF8   = powf(2.0f, -17.0f); // subnormal value
        minNormF8   = powf(2.0f, -14.0f); // min normal
    }
    // Use (maxT8 + 256) * scale_div[1]: exactly representable in bf16/fp32
    // fp8: (448+256)*2=1408 (exact in fp16/bf16/fp32)
    // bf8: (57344+256)*2=115200 (exact in bf16/fp32, exceeds fp16 max)
    float overflowF8 = std::is_same_v<T, fp16_t> ? maxT : (maxT8 + 256.f) * scale_div[1];
    float roundingF8 = 2.0625f;

    const auto test_data = std::array{2.0f,
                                      -4.0f,
                                      896.f,
                                      -896.f,
                                      qnanT,
                                      infT,
                                      underflowF8,
                                      roundingF8, /* set-1 with div < mul */
                                      -8.0f,
                                      16.0f,
                                      subNormF8,
                                      minNormF8,
                                      maxT,
                                      lowestT,
                                      overflowF8,
                                      -overflowF8 /* set-2 with div > mul */};

    constexpr int N = test_data.size();
    static_assert(test_data.size() % 8 == 0);
    static_assert(scale_div.size() == N / 8);
    static_assert(scale_mul.size() == N / 8);

    /* Expected results after: input/scale_div -> fp8/bf8 -> output*scale_mul */
    constexpr bool is_fp8 = std::is_same_v<TF8, fp8_t>;
    auto Inf_out          = (is_device) ? ((is_fp8) ? qnanT : infT) : (maxT8 * scale_mul[0]);
    auto minNorm_out      = (is_fp8) ? 1.0f : 0.00390625f;
    // device: fp8 - qnanT, bf8 round to InfT; host : max
    auto maxT_out = (is_device) ? ((is_fp8) ? qnanT : infT) : (maxT8 * scale_mul[1]);
    // device: fp8 - qnanT, bf8 round to max; host : max
    auto overflow_out =
        (std::is_same_v<T, fp16_t>)
            ? maxT_out
            : (is_fp8 ? (is_device ? qnanT : maxT8 * scale_mul[1]) : maxT8 * scale_mul[1]);

    const auto ref_data = std::array{
        /* Set-1 expected results (scale_div=4.0, scale_mul=2.0): */
        1.0f,    // [0] 2.0/4=0.5 -> fp8(0.5) -> 0.5*2=1.0
        -2.0f,   // [1] -4.0/4=-1.0 -> fp8(-1.0) -> -1.0*2=-2.0
        448.f,   // [2] 896/4=224 -> fp8/bf8(224) -> 224*2=448
        -448.f,  // [3] -896/4=-224 -> fp8/bf8(-224) -> -224*2=-448
        qnanT,   // [4] NaN -> fp8(NaN) -> NaN
        Inf_out, // [5] Inf
        0.0f,    // [6] underflowF8 2^-10/4=2^-12 -> fp8(0, below min subnormal) -> 0*2=0
        1.0f,    // [7] roundingF82.0625/4=0.515625 -> fp8(0.5) rounded -> 0.5*2=1.0
        /* Set-2 expected results (scale_div=2.0, scale_mul=128.0): */
        -512.f,       // [8]  -8/2=-4 -> fp8(-4) -> -4*128=-512
        1024.f,       // [9]  16/2=8 -> fp8(8) -> 8*128=1024
        0.0f,         // [10] subNormF8/2 -> fp8(0, below min subnormal) -> 0*128=0
        minNorm_out,  // [11] minNormT
        maxT_out,     // [12] maxT
        -maxT_out,    // [13] lowestT
        overflow_out, // [14] overflowF8
        -overflow_out // [15] overflowF8Neg
    };

    static_assert(test_data.size() == ref_data.size());

    std::array<T, N> in;
    std::array<T, N> ref, out;

    // prepare input and ground truth in host
    for(int i = 0; i < N; ++i)
    {
        in[i]  = type_convert<T>(test_data[i]);
        ref[i] = type_convert<T>(ref_data[i]);
    }

    using job = SrcPk8F8Dst<TF8, T, N>;

    if constexpr(is_device)
    {
        auto in_d   = std::make_unique<ck_tile::DeviceMem>(in.size() * sizeof(T));
        auto out_d  = std::make_unique<ck_tile::DeviceMem>(out.size() * sizeof(T));
        auto sdiv_d = std::make_unique<ck_tile::DeviceMem>(N * sizeof(float));
        auto smul_d = std::make_unique<ck_tile::DeviceMem>(N * sizeof(float));
        in_d->ToDevice(in.data());
        sdiv_d->ToDevice(scale_div.data());
        smul_d->ToDevice(scale_mul.data());

        MyKernel<job><<<1, 1>>>(reinterpret_cast<const T*>(in_d->GetDeviceBuffer()),
                                reinterpret_cast<T*>(out_d->GetDeviceBuffer()),
                                reinterpret_cast<const float*>(sdiv_d->GetDeviceBuffer()),
                                reinterpret_cast<const float*>(smul_d->GetDeviceBuffer()));

        out_d->FromDevice(out.data());
    }
    else
    {
        job{}(in.data(), out.data(), scale_div.data(), scale_mul.data());
    }

    for(int i = 0; i < N; ++i)
    {
        if(std::isnan(toF32(ref[i])))
            EXPECT_TRUE(std::isnan(toF32(out[i])))
                << "i:" << i << " expected: NaN, got:" << toF32(out[i]);
        else
            EXPECT_EQ(ref[i], out[i])
                << "i:" << i << " expected:" << toF32(ref[i]) << ", got:" << toF32(out[i]);
    }
}

/* Kernel for testing pkscale_type_convert with Packed4Scale */
template <typename Src, typename DST, int N, bool Block16Mod>
struct F8TestPkscaleTypeConvert
{
    CK_TILE_DEVICE void operator()([[maybe_unused]] float val,
                                   Packed4Scale_E8M0::raw_type* p_scale,
                                   DST* dst_data) const
    {
        if(dst_data == nullptr || p_scale == nullptr)
            return;

#if defined(__gfx125__)
        using DSTx8_t = ck_tile::ext_vector_t<DST, 8>;
        using Srcx8_t = ck_tile::ext_vector_t<Src, 8>;
        Srcx8_t in_f8(type_convert<Src>(val)); // assume unit matrix
        ck_tile::index_t lid = __lane_id();
        Packed4Scale_E8M0 scale(p_scale[lid]);

        constexpr int Nitr = (Block16Mod) ? 4 : 8;
        ck_tile::static_for<0, Nitr, 1>{}([&](auto it) {
            constexpr int opsel = (Block16Mod) ? (it + 8) : it;
            auto vT8 = ck_tile::pk4scaled_type_convert<DSTx8_t, Srcx8_t, opsel>(in_f8, scale);

            /* Row index of dst_data:
             * (lid & 0x0F): mapping lane0-15 and 16-31 to row 0-15
             * Column index of p_mat:
             *  it * 16: each iteration process 16 columns
             * ((lid >> 4) & 1) * 8: lane 0-15 write first 8 column
             *                       lane 16-31 write the next 8 columns*/
            ck_tile::static_for<0, 8, 1>{}([&](auto ii) {
                dst_data[(lid & 0x0F) * N + it * 16 + ((lid >> 4) & 1) * 8 + ii] =
                    vT8[static_cast<int>(ii)];
            });
        });
#endif
    }
};

template <typename Src, typename DST, bool Block16Mod>
void test_f8_pkscale_type_convert_device()
{
    // matrix shape M x N
    constexpr int M        = 16;
    constexpr int NBlock   = (Block16Mod) ? 8 : 16; // elements share a scale
    constexpr int N_scale  = 8;                     // 8 scale factors per row
    constexpr int N        = NBlock * N_scale;
    constexpr int mat_fval = 1.0f;
    std::vector<DST> out(M * N);

    int scale_init_option = 1; // 0: fixed value on the same column, 1: random values

    /* From a float scale matrix [M * N_scale=8] to a packed-4 scale matrix [M * 2] */
    /*Each 8 elements share one scale factor
      n:       [0:7]      [8:15]     [16:23]    [24:31]    [32:39]    [40:47]    [48:55]    [56:63]
      index:   0          1          2          3          4          5          6          7
      m[0:15]  fscale[m][0] ...
    */
    std::vector<float> fscale(M * N_scale);
    if(scale_init_option == 0)
    {
        // Option 0: Fixed pattern with wide dynamic range (same for all rows)
        for(int m = 0; m < M; m++)
        {
            fscale[m * N_scale + 0] = std::pow(2.0f, -10.0f); // 2^-10 ≈ 0.000977
            fscale[m * N_scale + 1] = std::pow(2.0f, -5.0f);  // 2^-5  = 0.03125
            fscale[m * N_scale + 2] = std::pow(2.0f, 8.0f);   // 2^8   = 256
            fscale[m * N_scale + 3] = std::pow(2.0f, 16.0f);  // 2^16  = 65536
            fscale[m * N_scale + 4] = std::pow(2.0f, 2.0f);   // 2^2   = 4
            fscale[m * N_scale + 5] = std::pow(2.0f, 4.0f);   // 2^4   = 16
            fscale[m * N_scale + 6] = std::pow(2.0f, -2.0f);  // 2^-2  = 0.25
            fscale[m * N_scale + 7] = std::pow(2.0f, 12.0f);  // 2^12  = 4096
        }
    }
    else if(scale_init_option == 1)
    {
        // Option 1: Random scales - each row gets different random power-of-2 values
        std::srand(42); // Fixed seed for reproducibility
        for(int m = 0; m < M; m++)
        {
            for(int s = 0; s < N_scale; s++)
            {
                // Random exponent in range [-20, 20] for wide dynamic range
                int exponent            = (std::rand() % 41) - 20;
                fscale[m * N_scale + s] = std::pow(2.0f, static_cast<float>(exponent));
            }
        }
    }

    std::vector<Packed4Scale_E8M0::raw_type> scale(2 * M);
    for(int m = 0; m < M; m++)
    {
        if constexpr(Block16Mod)
        {
            /* Each iteration take care of 16 x [8 + 8] matrix
             * opsel-8, use scale[th0:15]   [7:0]->th0:15, [15:8]->th16:32
             * opsel-9, use scale[th16:31]  [7:0]->th0:15, [15:8]->th16:32
             * opsel-10, use scale[th0:15]   [23:16]->th0:15, [31:24]->th16:32
             * opsel-11, use scale[th16:31]  [23:16]->th0:15, [31:24]->th16:32 */
            Packed4Scale_E8M0 scale4(fscale[m * N_scale + 5],
                                     fscale[m * N_scale + 4],
                                     fscale[m * N_scale + 1],
                                     fscale[m * N_scale + 0]);
            scale[m] = scale4.data(); // will load by th0-15
            scale4.set_scales_from_float(fscale[m * N_scale + 7],
                                         fscale[m * N_scale + 6],
                                         fscale[m * N_scale + 3],
                                         fscale[m * N_scale + 2]);
            scale[m + M] = scale4.data(); // will load by th16-31
        }
        else
        { // Block32Mod
            /* Each iteration take care of 16 x [8 + 8] matrix
             * opsel-0, use scale[th0:15]   [7:0]  ->th0:15 and th16:32
             * opsel-1, use scale[th23:16]  [7:0]  ->th0:15 and th16:32
             * opsel-2, use scale[th0:15]   [23:16]->th0:15 and th16:32
             * opsel-3, use scale[th23:16]  [23:16]->th0:15 and th16:32
             * opsel-4, use scale[th0:15]   [15:8] ->th0:15 and th16:32
             * opsel-5, use scale[th23:16]  [15:8] ->th0:15 and th16:32
             * opsel-6, use scale[th0:15]   [31:24]->th0:15 and th16:32
             * opsel-7, use scale[th23:16]  [31:24]->th0:15 and th16:32 */
            Packed4Scale_E8M0 scale4(fscale[m * N_scale + 6],
                                     fscale[m * N_scale + 2],
                                     fscale[m * N_scale + 4],
                                     fscale[m * N_scale + 0]);
            scale[m] = scale4.data(); // will load by th0-15
            scale4.set_scales_from_float(fscale[m * N_scale + 7],
                                         fscale[m * N_scale + 3],
                                         fscale[m * N_scale + 5],
                                         fscale[m * N_scale + 1]);
            scale[m + M] = scale4.data(); // will load by th16-31
        }
    }

    ck_tile::DeviceMem device_out(M * N * sizeof(DST));
    ck_tile::DeviceMem device_scale(2 * M * sizeof(Packed4Scale_E8M0::raw_type));
    device_scale.ToDevice(scale.data());

    MyKernel<F8TestPkscaleTypeConvert<Src, DST, N, Block16Mod>>
        <<<1, 32>>>(mat_fval,
                    reinterpret_cast<Packed4Scale_E8M0::raw_type*>(device_scale.GetDeviceBuffer()),
                    reinterpret_cast<DST*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* verify */
    for(int m = 0; m < M; m++)
    {
        for(int scale_idx = 0; scale_idx < N_scale; scale_idx++)
        {
            float expected_scale = fscale[m * N_scale + scale_idx];
            for(int n = scale_idx * NBlock; n < (scale_idx + 1) * NBlock; n++)
            {
                EXPECT_EQ(out[m * N + n], ck_tile::type_convert<DST>(expected_scale * mat_fval))
                    << "m: " << m << ", n: " << n << ", scale_idx: " << scale_idx
                    << ", expected: " << (expected_scale * mat_fval) << std::endl;
            }
        }
    }
}
