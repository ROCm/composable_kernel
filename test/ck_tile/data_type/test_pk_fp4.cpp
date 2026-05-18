// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using ck_tile::bf16_t;
using ck_tile::bf16x2_t;
using ck_tile::bf16x8_t;
using ck_tile::fp16_t;
using ck_tile::fp16x2_t;
using ck_tile::fp16x8_t;
using ck_tile::fp32_t;
using ck_tile::fp32x2_t;
using ck_tile::fp32x8_t;
using ck_tile::number;
using ck_tile::Packed4Scale_E8M0;
using ck_tile::pk_fp4_t;
using ck_tile::pk_fp4x4_t;

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_convert();

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_scaled_convert();

template <typename DST, bool Block16Mod = false>
CK_TILE_HOST void test_pkscale_type_convert_device();

TEST(PackedFp4, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::has_inf(), false);
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::zero(), pk_fp4_t{0b00000000});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::min(), pk_fp4_t{0b00100010});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::max(), pk_fp4_t{0b01110111});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::lowest(), pk_fp4_t{0b11111111});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::epsilon(), pk_fp4_t{0b00010001});
    EXPECT_EQ(ck_tile::numeric<pk_fp4_t>::round_error(), pk_fp4_t{0b00010001});
}
TEST(PackedFp4, fill)
{
    std::vector<pk_fp4_t> v_fp4(4);
    ck_tile::FillUniformDistribution<pk_fp4_t>{1.f, 1.f}(v_fp4);
    EXPECT_EQ(v_fp4[0].get(), pk_fp4_t{0b00100010}.get());
}
TEST(PackedFp4, ConvertBasic)
{
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(0.0f), pk_fp4_t{0b00000000}.get());
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(-0.0f), pk_fp4_t{0b00001000}.get());
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp4_t>(-1.0f), pk_fp4_t{0b00001010}.get());
    EXPECT_EQ(ck_tile::type_convert<pk_fp4_t>(0.0f), pk_fp4_t{0b00000000});
    EXPECT_EQ(ck_tile::type_convert<pk_fp4_t>(-0.0f), pk_fp4_t{0b00001000});
    EXPECT_EQ(ck_tile::type_convert<pk_fp4_t>(-1.0f), pk_fp4_t{0b00001010});
    EXPECT_EQ(pk_fp4_t(0.0f), pk_fp4_t{0b00000000});
    EXPECT_EQ(pk_fp4_t(-0.0f), pk_fp4_t{0b00001000});
    EXPECT_EQ(pk_fp4_t(-1.0f), pk_fp4_t{0b00001010});
    EXPECT_EQ(pk_fp4_t{0.0f}, pk_fp4_t{0b00000000});
    EXPECT_EQ(pk_fp4_t{-0.0f}, pk_fp4_t{0b00001000});
    EXPECT_EQ(pk_fp4_t{-1.0f}, pk_fp4_t{0b00001010});
}
TEST(PackedFp4, NumericBasic)
{
    auto f1  = pk_fp4_t{1.5f};
    auto f2  = pk_fp4_t{3.0f};
    auto ref = pk_fp4_t{-1.5f};
    EXPECT_EQ(f1 - f2, ref);
}
TEST(PackedFp4, ConvertDevice)
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
TEST(PackedFp4, ConvertHost)
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

TEST(PackedFp4, ScaledConvertDevice)
{
    constexpr bool is_device = true;
    test_scaled_convert<fp32_t, fp32_t, is_device>(); // fp32x8 -> fp4 -> fp32x8
    test_scaled_convert<fp16_t, fp16_t, is_device>();
    test_scaled_convert<bf16_t, bf16_t, is_device>();
    test_scaled_convert<fp32_t, fp16_t, is_device>();
    test_scaled_convert<fp32_t, bf16_t, is_device>();
    test_scaled_convert<fp16_t, fp32_t, is_device>();
    test_scaled_convert<bf16_t, fp32_t, is_device>();
}

TEST(PackedFp4, ScaledConvertHost)
{
    constexpr bool is_device = false;
    test_scaled_convert<fp32_t, fp32_t, is_device>(); // fp32x8 -> fp4 -> fp32x8
    test_scaled_convert<fp16_t, fp16_t, is_device>();
    test_scaled_convert<bf16_t, bf16_t, is_device>();
    test_scaled_convert<fp32_t, fp16_t, is_device>();
    test_scaled_convert<fp32_t, bf16_t, is_device>();
    test_scaled_convert<fp16_t, fp32_t, is_device>();
    test_scaled_convert<bf16_t, fp32_t, is_device>();
}

TEST(PackedFp4, PkscaleTypeConvertOpsel0_3)
{
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }
    test_pkscale_type_convert_device<fp32_t>();
    test_pkscale_type_convert_device<fp16_t>();
    test_pkscale_type_convert_device<bf16_t>();
}

TEST(PackedFp4, PkscaleTypeConvertOpsel4_7)
{
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }

    if(ck_tile::get_device_revision() == 0)
    {
        // Block16 Mode here means scale option [4-7].
        GTEST_SKIP() << "Block16 Mode not supported on asicRevision=0";
    }
    test_pkscale_type_convert_device<fp32_t, true>();
    test_pkscale_type_convert_device<fp16_t, true>();
    test_pkscale_type_convert_device<bf16_t, true>();
}

#define toF32(x) ck_tile::type_convert<float>(x)
#define toPF4(x) ck_tile::type_convert<pk_fp4_t>(x)
#define toSRC(x) ck_tile::type_convert<SRC>(x)
#define toDST(x) ck_tile::type_convert<DST>(x)
#define toDSTx2(x) ck_tile::type_convert<DSTx2_t>(x)

template <typename Kernel, typename... Args>
__global__ void MyKernel(Args... args)
{
    Kernel{}(args...);
}
template <typename SRC, typename DST, int N>
struct SrcPkfp4Dst
{
    CK_TILE_HOST_DEVICE void operator()(const SRC* src, DST* dst) const
    {

        using SRCx2_t = ck_tile::ext_vector_t<SRC, 2>;
        using DSTx2_t = ck_tile::ext_vector_t<DST, 2>;

        ck_tile::static_for<0, N, 2>{}([&](auto i) {
            const auto input2 = SRCx2_t{src[i], src[i + 1]};

            if(i % 4 == 0)
            {
                // ex: fp32_t -> fp4 -> bf16_t
                dst[i] = toDST(toPF4(src[i]));
                // ex: fp32x2_t -> pk_fp4 -> unpack<0> -> bf16_t
                dst[i + 1] = toDST(toPF4(input2).unpack(number<1>{}));
            }
            else
            {
                // ex: fp32x2_t -> pk_fp4_t -> bf16x2_t
                reinterpret_cast<DSTx2_t*>(dst)[i >> 1] = toDSTx2(toPF4(input2));
            }
        });
    }
};

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_convert()
{
    const auto test_data = std::array{0.f,  0.25f,  0.5f,  0.75f,  1.f,  1.25f,  1.5f,    1.75f,
                                      -0.f, -0.25f, -0.5f, -0.75f, -1.f, -1.25f, -1.5f,   -1.75f,
                                      2.f,  2.5f,   3.f,   3.5f,   4.f,  5.f,    5.0625f, 6.f};
    const auto ref_data =
        std::array{0.f,  0.f,  0.5f,  1.f,  1.f, 1.f, 1.5f, 2.f, -0.f, -0.f, -0.5f, -1.f,
                   -1.f, -1.f, -1.5f, -2.f, 2.f, 2.f, 3.f,  4.f, 4.f,  4.f,  6.f,   6.f};

    static_assert(test_data.size() == ref_data.size());
    static_assert(test_data.size() % 2 == 0);

    constexpr int N = test_data.size();
    std::array<SRC, N> in;
    std::array<DST, N> ref, out;

    // prepare input and ground truth in host
    for(int i = 0; i < N; ++i)
    {
        in[i]  = toSRC(test_data[i]);
        ref[i] = toDST(ref_data[i]);
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
                                reinterpret_cast<DST*>(out_d->GetDeviceBuffer()));

        out_d->FromDevice(out.data());
    }
    else
    {
        job{}(in.data(), out.data());
    }

    for(int i = 0; i < N; ++i)
        EXPECT_EQ(ref[i], out[i]) << "i:" << i;
}

/* Kernel for testing 8-element vector conversions with scaling */
template <typename SRC, typename DST, int N>
struct SrcScaledPkfp4x8Dst
{
    CK_TILE_HOST_DEVICE void operator()(const SRC* src, DST* dst, float scale) const
    {
        using SRCx8_t = ck_tile::ext_vector_t<SRC, 8>;
        using DSTx8_t = ck_tile::ext_vector_t<DST, 8>;

        ck_tile::static_for<0, N, 8>{}([&](auto i) {
            SRCx8_t input8{};
            for(int j = 0; j < 8; ++j)
                input8[j] = src[i + j];

            // Convert: SRCx8 -> pk_fp4x4_t -> DSTx8 with scaling
            auto fp4_packed = ck_tile::scaled_type_convert<pk_fp4x4_t>(input8, scale);
            DSTx8_t output8 = ck_tile::scaled_type_convert<DSTx8_t>(fp4_packed, scale);

            for(int j = 0; j < 8; ++j)
                dst[i + j] = output8[j];
        });
    }
};

template <typename SRC, typename DST, bool is_device>
CK_TILE_HOST void test_scaled_convert()
{
    constexpr float scale = 2.0f;

    const auto test_data = std::array{
        0.f, 0.5f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 0.125f, 0.75f, 1.5f, 2.5f, 10.f, 12.f, 14.f, 16.f};

    /* Expected results after: input/scale -> FP4 quantize -> output*scale
     * For scale=2.0:
     *   0.0/2=0.0 -> 0   -> 0.0*2 = 0.0
     *   0.5/2=0.25 -> 0  -> 0.0*2 = 0.0
     *   1.0/2=0.5 -> 0.5 -> 0.5*2 = 1.0
     *   2.0/2=1.0 -> 1   -> 1.0*2 = 2.0
     *   3.0/2=1.5 -> 1.5 -> 1.5*2 = 3.0
     *   4.0/2=2.0 -> 2   -> 2.0*2 = 4.0
     *   6.0/2=3.0 -> 3   -> 3.0*2 = 6.0
     *   8.0/2=4.0 -> 4   -> 4.0*2 = 8.0
     *   0.125/2=0.0625 -> 0 -> 0.0*2 = 0.0
     *   0.75/2=0.375 -> 0.5(rne) -> 0.5*2 = 1.0
     *   1.5/2=0.75 -> 1(rne)   -> 1.0*2 = 2.0
     *   2.5/2=1.25 -> 1(rne)   -> 1.0*2 = 2.0
     *   10.0/2=5.0 -> 4(rne)   -> 4.0*2 = 8.0
     *   12.0/2=6.0 -> 6   -> 6.0*2 = 12.0 (max representable with scale=2)
     *   14.0/2=7.0 -> 6 (clamp) -> 6.0*2 = 12.0
     *   16.0/2=8.0 -> 6 (clamp) -> 6.0*2 = 12.0 */
    const auto ref_data = std::array{
        0.f, 0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 0.f, 1.f, 2.f, 2.f, 8.f, 12.f, 12.f, 12.f};

    static_assert(test_data.size() == ref_data.size());
    static_assert(test_data.size() % 8 == 0);

    constexpr int N = test_data.size();
    std::array<SRC, N> in;
    std::array<DST, N> ref, out;

    for(int i = 0; i < N; ++i)
    {
        in[i]  = toSRC(test_data[i]);
        ref[i] = toDST(ref_data[i]);
        EXPECT_EQ(test_data[i], toF32(in[i]));
        EXPECT_EQ(ref_data[i], toF32(ref[i]));
    }

    using job = SrcScaledPkfp4x8Dst<SRC, DST, N>;

    if constexpr(is_device)
    {
        auto in_d  = std::make_unique<ck_tile::DeviceMem>(in.size() * sizeof(SRC));
        auto out_d = std::make_unique<ck_tile::DeviceMem>(out.size() * sizeof(DST));
        in_d->ToDevice(in.data());

        MyKernel<job><<<1, 1>>>(reinterpret_cast<const SRC*>(in_d->GetDeviceBuffer()),
                                reinterpret_cast<DST*>(out_d->GetDeviceBuffer()),
                                scale);

        out_d->FromDevice(out.data());
    }
    else
    {
        job{}(in.data(), out.data(), scale);
    }

    for(int i = 0; i < N; ++i)
        EXPECT_EQ(ref[i], out[i]) << "i:" << i << " expected:" << toF32(ref[i])
                                  << " got:" << toF32(out[i]);
}

/* Kernel for testing pkscale_type_convert with Packed4Scale */
template <typename DST, int N, bool Block16Mod>
struct TestPkscaleTypeConvert
{
    CK_TILE_DEVICE void operator()([[maybe_unused]] pk_fp4x4_t val,
                                   Packed4Scale_E8M0::raw_type* p_scale,
                                   DST* dst_data) const
    {
        if(dst_data == nullptr || p_scale == nullptr)
            return;

#if defined(__gfx125__)
        using DSTx8_t        = ck_tile::ext_vector_t<DST, 8>;
        ck_tile::index_t lid = __lane_id();
        Packed4Scale_E8M0 scale(p_scale[lid]);

        ck_tile::static_for<0, 4, 1>{}([&](auto it) {
            constexpr int opsel = (Block16Mod) ? (it + 4) : it;
            auto vT8 = ck_tile::pk4scaled_type_convert<DSTx8_t, pk_fp4x4_t, opsel>(val, scale);

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

template <typename DST, bool Block16Mod>
void test_pkscale_type_convert_device()
{
    // matrix shape M x N
    constexpr int M        = 16;
    constexpr int N        = 64;
    constexpr int N_scale  = N / 8; // every 8 elements share a scale in packed-8 type convert
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
             * opsel-4, use scale[th0:15]   [7:0]->th0:15, [23:16]->th16:32
             * opsel-5, use scale[th16:31]  [7:0]->th0:15, [23:16]->th16:32
             * opsel-6, use scale[th0:15]   [15:8]->th0:15, [31:24]->th16:32
             * opsel-7, use scale[th16:31]  [15:8]->th0:15, [31:24]->th16:32 */
            Packed4Scale_E8M0 scale4(fscale[m * N_scale + 5],
                                     fscale[m * N_scale + 1],
                                     fscale[m * N_scale + 4],
                                     fscale[m * N_scale + 0]);
            scale[m] = scale4.data(); // will load by th0-15
            scale4.set_scales_from_float(fscale[m * N_scale + 7],
                                         fscale[m * N_scale + 3],
                                         fscale[m * N_scale + 6],
                                         fscale[m * N_scale + 2]);
            scale[m + M] = scale4.data(); // will load by th16-31
        }
        else
        { // Block32Mod
            /* Each iteration take care of 16 x [8 + 8] matrix
             * opsel-0, use scale[th0:15]   [7:0]->th0:15, [15:8]->th16:32
             * opsel-1, use scale[th16:31]  [7:0]->th0:15, [15:8]->th16:32
             * opsel-2, use scale[th0:15]   [23:16]->th0:15, [31:24]->th16:32
             * opsel-3, use scale[th16:31]  [23:16]->th0:15, [31:24]->th16:32 */
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
    }

    /* Simplified here with matrix filled with mat_fval
     * pack 4 float data to pk_fp4x4_t */
    pk_fp4_t pkf4_val = ck_tile::type_convert<pk_fp4_t>(fp32x2_t(mat_fval));
    auto make_pk_fp4x4_from_pk_fp4 =
        [](pk_fp4_t p0, pk_fp4_t p1, pk_fp4_t p2, pk_fp4_t p3) -> pk_fp4x4_t {
        return pk_fp4x4_t{p0.get(), p1.get(), p2.get(), p3.get()};
    };
    pk_fp4x4_t pkf4x4_val = make_pk_fp4x4_from_pk_fp4(pkf4_val, pkf4_val, pkf4_val, pkf4_val);

    ck_tile::DeviceMem device_out(M * N * sizeof(DST));
    ck_tile::DeviceMem device_scale(2 * M * sizeof(Packed4Scale_E8M0::raw_type));
    device_scale.ToDevice(scale.data());

    MyKernel<TestPkscaleTypeConvert<DST, N, Block16Mod>>
        <<<1, 32>>>(pkf4x4_val,
                    reinterpret_cast<Packed4Scale_E8M0::raw_type*>(device_scale.GetDeviceBuffer()),
                    reinterpret_cast<DST*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* verify */
    for(int m = 0; m < M; m++)
    {
        for(int scale_idx = 0; scale_idx < N_scale; scale_idx++)
        {
            float expected_scale = fscale[m * N_scale + scale_idx];
            for(int n = scale_idx * 8; n < (scale_idx + 1) * 8; n++)
            {
                EXPECT_EQ(out[m * N + n], ck_tile::type_convert<DST>(expected_scale * mat_fval))
                    << "m: " << m << ", n: " << n << ", scale_idx: " << scale_idx
                    << ", expected: " << (expected_scale * mat_fval) << std::endl;
            }
        }
    }
}
