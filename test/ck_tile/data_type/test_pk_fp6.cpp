// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <vector>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/host.hpp"

using ck_tile::bf16_t;
using ck_tile::bf16x16_t;
using ck_tile::fp16_t;
using ck_tile::fp16x16_t;
using ck_tile::fp32_t;
using ck_tile::fp32x16_t;
using ck_tile::fp32x8_t;
using ck_tile::number;
using ck_tile::pk_bf6_t;
using ck_tile::pk_fp6_t;

template <
    typename SRC,
    typename PK6,
    typename DST,
    bool is_device,
    std::enable_if_t<std::is_same_v<PK6, pk_fp6_t> || std::is_same_v<PK6, pk_bf6_t>, bool> = true>
CK_TILE_HOST void test_convert();

template <
    typename SRC,
    typename PK6,
    typename DST,
    bool is_device,
    std::enable_if_t<std::is_same_v<PK6, pk_fp6_t> || std::is_same_v<PK6, pk_bf6_t>, bool> = true>
CK_TILE_HOST void test_scaled_convert();

template <typename PK6, typename DST, bool Block16Mod = false>
CK_TILE_HOST void test_pkscale_type_convert_device();

// ============================================================================
// FP6 (E2M3) Tests
// ============================================================================

TEST(PackedFp6, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<pk_fp6_t>::has_inf(), false);

    // FP6 E2M3: bias=1, range ~[0.125, 7.0]
    // Test using the binary constants directly
    pk_fp6_t zero_pk        = ck_tile::numeric<pk_fp6_t>::zero();
    pk_fp6_t min_pk         = ck_tile::numeric<pk_fp6_t>::min();
    pk_fp6_t max_pk         = ck_tile::numeric<pk_fp6_t>::max();
    pk_fp6_t lowest_pk      = ck_tile::numeric<pk_fp6_t>::lowest();
    pk_fp6_t epsilon_pk     = ck_tile::numeric<pk_fp6_t>::epsilon();
    pk_fp6_t round_error_pk = ck_tile::numeric<pk_fp6_t>::round_error();
    pk_fp6_t denorm_min_pk  = ck_tile::numeric<pk_fp6_t>::denorm_min();
    EXPECT_FLOAT_EQ(zero_pk.to_float(1.0f), 0.0f);
    EXPECT_FLOAT_EQ(min_pk.to_float(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(max_pk.to_float(1.0f), 7.5f);
    EXPECT_FLOAT_EQ(lowest_pk.to_float(1.0f), -7.5f);
    EXPECT_FLOAT_EQ(epsilon_pk.to_float(1.0f), 0.125f);
    EXPECT_FLOAT_EQ(round_error_pk.to_float(1.0f), 0.125f);
    EXPECT_FLOAT_EQ(denorm_min_pk.to_float(1.0f), 0.125f);
}

TEST(PackedFp6, Fill)
{
    std::vector<pk_fp6_t> v_fp6(2);
    ck_tile::FillUniformDistribution<pk_fp6_t>{1.f, 1.f}(v_fp6);
    pk_fp6_t expected;
    expected.set_element(0, 0b001000);
    EXPECT_EQ(v_fp6[0].get_element(0), expected.get_element(0));
    EXPECT_EQ(v_fp6[0].get_element(6), expected.get_element(0));
    EXPECT_EQ(v_fp6[1].get_element(15), expected.get_element(0));
}

TEST(PackedFp6, ConvertBasic)
{
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp6_t>(0.0f), 0b000000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp6_t>(-0.0f), 0b100000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp6_t>(1.0f), 0b001000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp6_t>(-1.0f), 0b101000);

    EXPECT_EQ(ck_tile::type_convert<pk_fp6_t>(0.0f).get_element(0), 0b000000);
    EXPECT_EQ(ck_tile::type_convert<pk_fp6_t>(-0.0f).get_element(0), 0b100000);
    EXPECT_EQ(ck_tile::type_convert<pk_fp6_t>(1.0f).get_element(0), 0b001000);
    EXPECT_EQ(ck_tile::type_convert<pk_fp6_t>(-1.0f).get_element(0), 0b101000);

    EXPECT_EQ(pk_fp6_t(0.0f).get_element(0), 0b000000);
    EXPECT_EQ(pk_fp6_t(-0.0f).get_element(0), 0b100000);
    EXPECT_EQ(pk_fp6_t(1.0f).get_element(0), 0b001000);
    EXPECT_EQ(pk_fp6_t(-1.0f).get_element(0), 0b101000);
}

TEST(PackedFp6, ConvertHost)
{
    constexpr bool is_device = false;
    test_convert<fp32_t, pk_fp6_t, fp32_t, is_device>();
    test_convert<fp16_t, pk_fp6_t, fp16_t, is_device>();
    test_convert<bf16_t, pk_fp6_t, bf16_t, is_device>();
    test_convert<fp32_t, pk_fp6_t, fp16_t, is_device>();
    test_convert<fp32_t, pk_fp6_t, bf16_t, is_device>();
    test_convert<fp16_t, pk_fp6_t, fp32_t, is_device>();
    test_convert<bf16_t, pk_fp6_t, fp32_t, is_device>();
}

TEST(PackedFp6, ConvertDevice)
{
    constexpr bool is_device = true;
    test_convert<fp32_t, pk_fp6_t, fp32_t, is_device>();
    test_convert<fp16_t, pk_fp6_t, fp16_t, is_device>();
    test_convert<bf16_t, pk_fp6_t, bf16_t, is_device>();
    test_convert<fp32_t, pk_fp6_t, fp16_t, is_device>();
    test_convert<fp32_t, pk_fp6_t, bf16_t, is_device>();
    test_convert<fp16_t, pk_fp6_t, fp32_t, is_device>();
    test_convert<bf16_t, pk_fp6_t, fp32_t, is_device>();
}

TEST(PackedFp6, ScaledConvertHost)
{
    constexpr bool is_device = false;
    test_scaled_convert<fp32_t, pk_fp6_t, fp32_t, is_device>();
    test_scaled_convert<fp16_t, pk_fp6_t, fp16_t, is_device>();
    test_scaled_convert<bf16_t, pk_fp6_t, bf16_t, is_device>();
    test_scaled_convert<fp32_t, pk_fp6_t, fp16_t, is_device>();
    test_scaled_convert<fp32_t, pk_fp6_t, bf16_t, is_device>();
    test_scaled_convert<fp16_t, pk_fp6_t, fp32_t, is_device>();
    test_scaled_convert<bf16_t, pk_fp6_t, fp32_t, is_device>();
}

TEST(PackedFp6, ScaledConvertDevice)
{
    constexpr bool is_device = true;
    test_scaled_convert<fp32_t, pk_fp6_t, fp32_t, is_device>();
    test_scaled_convert<fp16_t, pk_fp6_t, fp16_t, is_device>();
    test_scaled_convert<bf16_t, pk_fp6_t, bf16_t, is_device>();
    test_scaled_convert<fp32_t, pk_fp6_t, fp16_t, is_device>();
    test_scaled_convert<fp32_t, pk_fp6_t, bf16_t, is_device>();
    test_scaled_convert<fp16_t, pk_fp6_t, fp32_t, is_device>();
    test_scaled_convert<bf16_t, pk_fp6_t, fp32_t, is_device>();
}

TEST(PackedFp6, PkscaleTypeConvertOpsel0_3)
{
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }
    test_pkscale_type_convert_device<pk_fp6_t, fp32_t>();
    test_pkscale_type_convert_device<pk_fp6_t, fp16_t>();
    test_pkscale_type_convert_device<pk_fp6_t, bf16_t>();
}

TEST(PackedFp6, PkscaleTypeConvertOpsel4_7)
{
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }
    test_pkscale_type_convert_device<pk_fp6_t, fp32_t, true>();
    test_pkscale_type_convert_device<pk_fp6_t, fp16_t, true>();
    test_pkscale_type_convert_device<pk_fp6_t, bf16_t, true>();
}

// ============================================================================
// BF6 (E3M2) Tests
// ============================================================================

TEST(PackedBf6, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<pk_bf6_t>::has_inf(), false);

    // BF6 E3M2: bias=3, range ~[0.25, 28.0]
    // Test using the binary constants directly
    pk_bf6_t zero_pk        = ck_tile::numeric<pk_bf6_t>::zero();
    pk_bf6_t min_pk         = ck_tile::numeric<pk_bf6_t>::min();
    pk_bf6_t max_pk         = ck_tile::numeric<pk_bf6_t>::max();
    pk_bf6_t lowest_pk      = ck_tile::numeric<pk_bf6_t>::lowest();
    pk_bf6_t epsilon_pk     = ck_tile::numeric<pk_bf6_t>::epsilon();
    pk_bf6_t round_error_pk = ck_tile::numeric<pk_bf6_t>::round_error();
    pk_bf6_t denorm_min_pk  = ck_tile::numeric<pk_bf6_t>::denorm_min();

    EXPECT_FLOAT_EQ(zero_pk.to_float(1.0f), 0.0f);
    EXPECT_FLOAT_EQ(min_pk.to_float(1.0f), 0.25f);
    EXPECT_FLOAT_EQ(max_pk.to_float(1.0f), 28.0f);
    EXPECT_FLOAT_EQ(lowest_pk.to_float(1.0f), -28.0f);
    EXPECT_FLOAT_EQ(epsilon_pk.to_float(1.0f), 0.0625f);
    EXPECT_FLOAT_EQ(round_error_pk.to_float(1.0f), 0.0625f);
    EXPECT_FLOAT_EQ(denorm_min_pk.to_float(1.0f), 0.0625f);
}

TEST(PackedBf6, Fill)
{
    std::vector<pk_bf6_t> v_bf6(2);
    ck_tile::FillUniformDistribution<pk_bf6_t>{1.f, 1.f}(v_bf6);
    pk_bf6_t expected;
    // 1.0f in BF6 E3M2: sign=0, exp=011, mant=00 = 0b001100
    expected.set_element(0, 0b001100);
    EXPECT_EQ(v_bf6[0].get_element(0), expected.get_element(0));
    EXPECT_EQ(v_bf6[0].get_element(6), expected.get_element(0));
    EXPECT_EQ(v_bf6[1].get_element(15), expected.get_element(0));
}

TEST(PackedBf6, ConvertBasic)
{
    // Test basic float to bf6 conversion
    // BF6 E3M2 format: sign(1) + exp(3) + mant(2)
    // 0.0f:  0 000 00 = 0b000000
    // -0.0f: 1 000 00 = 0b100000
    // 1.0f:  0 011 00 = 0b001100
    // -1.0f: 1 011 00 = 0b101100

    EXPECT_EQ(ck_tile::convert_to_type<pk_bf6_t>(0.0f), 0b000000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_bf6_t>(-0.0f), 0b100000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_bf6_t>(1.0f), 0b001100);
    EXPECT_EQ(ck_tile::convert_to_type<pk_bf6_t>(-1.0f), 0b101100);

    EXPECT_EQ(ck_tile::type_convert<pk_bf6_t>(0.0f).get_element(0), 0b000000);
    EXPECT_EQ(ck_tile::type_convert<pk_bf6_t>(-0.0f).get_element(0), 0b100000);
    EXPECT_EQ(ck_tile::type_convert<pk_bf6_t>(1.0f).get_element(0), 0b001100);
    EXPECT_EQ(ck_tile::type_convert<pk_bf6_t>(-1.0f).get_element(0), 0b101100);

    EXPECT_EQ(pk_bf6_t(0.0f).get_element(0), 0b000000);
    EXPECT_EQ(pk_bf6_t(-0.0f).get_element(0), 0b100000);
    EXPECT_EQ(pk_bf6_t(1.0f).get_element(0), 0b001100);
    EXPECT_EQ(pk_bf6_t(-1.0f).get_element(0), 0b101100);
}

TEST(PackedBf6, ConvertHost)
{
    constexpr bool is_device = false;
    test_convert<fp32_t, pk_bf6_t, fp32_t, is_device>();
    test_convert<fp16_t, pk_bf6_t, fp16_t, is_device>();
    test_convert<bf16_t, pk_bf6_t, bf16_t, is_device>();
    test_convert<fp32_t, pk_bf6_t, fp16_t, is_device>();
    test_convert<fp32_t, pk_bf6_t, bf16_t, is_device>();
    test_convert<fp16_t, pk_bf6_t, fp32_t, is_device>();
    test_convert<bf16_t, pk_bf6_t, fp32_t, is_device>();
}

TEST(PackedBf6, ConvertDevice)
{
    constexpr bool is_device = true;
    test_convert<fp32_t, pk_bf6_t, fp32_t, is_device>();
    test_convert<fp16_t, pk_bf6_t, fp16_t, is_device>();
    test_convert<bf16_t, pk_bf6_t, bf16_t, is_device>();
    test_convert<fp32_t, pk_bf6_t, fp16_t, is_device>();
    test_convert<fp32_t, pk_bf6_t, bf16_t, is_device>();
    test_convert<fp16_t, pk_bf6_t, fp32_t, is_device>();
    test_convert<bf16_t, pk_bf6_t, fp32_t, is_device>();
}

TEST(PackedBf6, ScaledConvertHost)
{
    constexpr bool is_device = false;
    test_scaled_convert<fp32_t, pk_bf6_t, fp32_t, is_device>();
    test_scaled_convert<fp16_t, pk_bf6_t, fp16_t, is_device>();
    test_scaled_convert<bf16_t, pk_bf6_t, bf16_t, is_device>();
    test_scaled_convert<fp32_t, pk_bf6_t, fp16_t, is_device>();
    test_scaled_convert<fp32_t, pk_bf6_t, bf16_t, is_device>();
    test_scaled_convert<fp16_t, pk_bf6_t, fp32_t, is_device>();
    test_scaled_convert<bf16_t, pk_bf6_t, fp32_t, is_device>();
}

TEST(PackedBf6, ScaledConvertDevice)
{
    constexpr bool is_device = true;
    test_scaled_convert<fp32_t, pk_bf6_t, fp32_t, is_device>();
    test_scaled_convert<fp16_t, pk_bf6_t, fp16_t, is_device>();
    test_scaled_convert<bf16_t, pk_bf6_t, bf16_t, is_device>();
    test_scaled_convert<fp32_t, pk_bf6_t, fp16_t, is_device>();
    test_scaled_convert<fp32_t, pk_bf6_t, bf16_t, is_device>();
    test_scaled_convert<fp16_t, pk_bf6_t, fp32_t, is_device>();
    test_scaled_convert<bf16_t, pk_bf6_t, fp32_t, is_device>();
}

TEST(PackedBf6, PkscaleTypeConvertOpsel0_3)
{
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }
    test_pkscale_type_convert_device<pk_bf6_t, fp32_t>();
    test_pkscale_type_convert_device<pk_bf6_t, fp16_t>();
    test_pkscale_type_convert_device<pk_bf6_t, bf16_t>();
}

TEST(PackedBf6, PkscaleTypeConvertOpsel4_7)
{
    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "Test for GFX1250.";
    }
    test_pkscale_type_convert_device<pk_bf6_t, fp32_t, true>();
    test_pkscale_type_convert_device<pk_bf6_t, fp16_t, true>();
    test_pkscale_type_convert_device<pk_bf6_t, bf16_t, true>();
}

// ============================================================================
// Cross-word boundary tests (for 6-bit packing)
// ============================================================================

TEST(PackedFp6, CrossWordBoundary)
{
    // Test elements that span across uint32_t boundaries
    // Element at bit position 5 (5*6 = 30 bits) spans words 0 and 1
    pk_fp6_t val;

    // Elements that might span boundaries
    val.set_element(5, 0b001010);  // bit offset 30, spans word 0-1
    val.set_element(10, 0b001100); // bit offset 60, spans word 1-2
    val.set_element(15, 0b010000); // bit offset 90, spans word 2-3

    EXPECT_EQ(val.unpack(number<5>{}), 0b001010);
    EXPECT_EQ(val.unpack(number<10>{}), 0b001100);
    EXPECT_EQ(val.unpack(number<15>{}), 0b010000);
}

TEST(PackedBf6, CrossWordBoundary)
{
    // Test elements that span across uint32_t boundaries
    pk_bf6_t val;

    val.set_element(5, 0b001010);
    val.set_element(10, 0b001100);
    val.set_element(15, 0b010000);

    EXPECT_EQ(val.unpack(number<5>{}), 0b001010);
    EXPECT_EQ(val.unpack(number<10>{}), 0b001100);
    EXPECT_EQ(val.unpack(number<15>{}), 0b010000);
}

// ============================================================================
// Implementation
// ============================================================================

#define toF32(x) ck_tile::type_convert<float>(x)
#define toSRC(x) ck_tile::type_convert<SRC>(x)
#define toDST(x) ck_tile::type_convert<DST>(x)

template <typename Kernel, typename... Args>
__global__ void MyKernel(Args... args)
{
    Kernel{}(args...);
}

/* Unified kernel for testing 16-element vector conversions with optional scaling */
template <typename SRC, typename PK6, typename DST, int N>
struct SrcPk6Dst
{
    CK_TILE_HOST_DEVICE void operator()(const SRC* src, DST* dst, float scale = 1.0f) const
    {
#define toPK6(x) ck_tile::scaled_type_convert<PK6>(x, scale)
#define toDSTx16(x) ck_tile::scaled_type_convert<DSTx16_t>(x, scale)
#if CK_TILE_AVX512F_WA
        // Use arrays of two 8-element vectors only for float to avoid AVX-512 on non-supporting
        // CPUs For smaller types (fp16, bf16), 16-element vectors are fine with AVX2
        constexpr bool UseSrcx8 = std::is_same_v<SRC, float>;
        constexpr bool UseDstx8 = std::is_same_v<DST, float>;

        using SRCx8_t  = ck_tile::ext_vector_t<SRC, 8>;
        using DSTx8_t  = ck_tile::ext_vector_t<DST, 8>;
        using SRCx16_t = std::conditional_t<UseSrcx8, SRCx8_t[2], ck_tile::ext_vector_t<SRC, 16>>;
        using DSTx16_t = std::conditional_t<UseDstx8, DSTx8_t[2], ck_tile::ext_vector_t<DST, 16>>;
#else
        // Use regular 16-element vectors when AVX-512 is available
        using SRCx16_t = ck_tile::ext_vector_t<SRC, 16>;
        using DSTx16_t = ck_tile::ext_vector_t<DST, 16>;
#endif

        ck_tile::static_for<0, N, 16>{}([&](auto i) {
#if CK_TILE_AVX512F_WA
            // Load input
            SRCx16_t input16{};
            if constexpr(UseSrcx8)
            {
                // Load as two 8-element vectors
                for(int j = 0; j < 8; ++j)
                {
                    input16[0][j] = src[i + j];
                    input16[1][j] = src[i + j + 8];
                }
            }
            else
            {
                // Load as single 16-element vector
                for(int j = 0; j < 16; ++j)
                    input16[j] = src[i + j];
            }

            // Convert: SRCx16 -> PK6 -> DSTx16 with scaling
            PK6 pk6_packed = toPK6(input16);
            DSTx16_t output16{};
            if constexpr(UseDstx8)
            {
                ck_tile::scaled_type_convert<DSTx8_t[2]>(pk6_packed, scale, output16);
            }
            else
            {
                output16 = toDSTx16(pk6_packed);
            }

            // Store output
            if constexpr(UseDstx8)
            {
                for(int j = 0; j < 8; ++j)
                {
                    dst[i + j]     = output16[0][j];
                    dst[i + j + 8] = output16[1][j];
                }
            }
            else
            {
                for(int j = 0; j < 16; ++j)
                    dst[i + j] = output16[j];
            }
#else
            // Standard 16-element vector path when AVX-512 is available
            SRCx16_t input16{};
            for(int j = 0; j < 16; ++j)
                input16[j] = src[i + j];

            PK6 pk6_packed    = toPK6(input16);
            DSTx16_t output16 = toDSTx16(pk6_packed);

            for(int j = 0; j < 16; ++j)
                dst[i + j] = output16[j];
#endif
        });
#undef toPK6
#undef toDSTx16
    }
};

template <typename SRC,
          typename PK6,
          typename DST,
          bool is_device,
          std::enable_if_t<std::is_same_v<PK6, pk_fp6_t> || std::is_same_v<PK6, pk_bf6_t>, bool>>
CK_TILE_HOST void test_convert()
{
    constexpr int N = 32;

    // FP6 E2M3 test values: bias=1, range [0.125, 7.5]
    constexpr std::array<float, N> fp6_test_data = {
        0.f,   0.125f, 0.25f, 0.375f, 0.5f,  0.625f, 0.75f, 0.875f, 1.f,  1.25f, 1.5f,
        1.75f, 2.f,    2.25f, 2.5f,   2.75f, 3.f,    3.5f,  4.f,    4.5f, 5.f,   5.5f,
        6.f,   6.5f,   7.f,   7.5f,   -1.f,  -2.f,   -3.f,  -5.f,   -7.f, -7.5f};
    // Expected values after FP6 quantization
    constexpr std::array<float, N> fp6_ref_data = {
        0.f,   0.125f, 0.25f, 0.375f, 0.5f,  0.625f, 0.75f, 0.875f, 1.f,  1.25f, 1.5f,
        1.75f, 2.f,    2.25f, 2.5f,   2.75f, 3.f,    3.5f,  4.f,    4.5f, 5.f,   5.5f,
        6.f,   6.5f,   7.f,   7.5f,   -1.f,  -2.f,   -3.f,  -5.f,   -7.f, -7.5f};

    // BF6 E3M2 test values: bias=3, range [0.0625, 28]
    constexpr std::array<float, N> bf6_test_data = {
        0.f,   0.0625f, 0.125f, 0.1875f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f, 1.f,
        1.25f, 1.5f,    1.75f,  2.f,     2.5f,  3.f,    3.5f, 4.f,    5.f,   6.f,    7.f,
        8.f,   10.f,    12.f,   14.f,    16.f,  24.f,   -1.f, -2.f,   -4.f,  -28.f};
    // Expected values after BF6 quantization
    constexpr std::array<float, N> bf6_ref_data = {
        0.f,   0.0625f, 0.125f, 0.1875f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f, 1.f,
        1.25f, 1.5f,    1.75f,  2.f,     2.5f,  3.f,    3.5f, 4.f,    5.f,   6.f,    7.f,
        8.f,   10.f,    12.f,   14.f,    16.f,  24.f,   -1.f, -2.f,   -4.f,  -28.f};

    // Select test data based on PK6 type
    const auto& test_data = (std::is_same_v<PK6, pk_fp6_t> ? fp6_test_data : bf6_test_data);
    const auto& ref_data  = (std::is_same_v<PK6, pk_fp6_t> ? fp6_ref_data : bf6_ref_data);

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

    using job = SrcPk6Dst<SRC, PK6, DST, N>;

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

template <typename SRC,
          typename PK6,
          typename DST,
          bool is_device,
          std::enable_if_t<std::is_same_v<PK6, pk_fp6_t> || std::is_same_v<PK6, pk_bf6_t>, bool>>
CK_TILE_HOST void test_scaled_convert()
{
    constexpr float scale = 2.0f;

    // FP6 E2M3 test values with scale=2.0: range [0.125, 7.5], scaled range [0.25, 15.0]
    constexpr std::array<float, 16> fp6_test_data = {0.f,
                                                     0.25f,
                                                     0.5f,
                                                     1.f,
                                                     2.f,
                                                     3.f,
                                                     4.f,
                                                     6.f,
                                                     0.125f,
                                                     0.75f,
                                                     1.5f,
                                                     2.5f,
                                                     8.f,
                                                     10.f,
                                                     14.f,
                                                     16.f};
    /* Expected results after: input/scale -> FP6 quantize -> output*scale
     * For scale=2.0:
     *   0.0/2=0.0    -> 0     -> 0.0*2 = 0.0
     *   0.25/2=0.125 -> 0.125 -> 0.125*2 = 0.25
     *   0.5/2=0.25   -> 0.25  -> 0.25*2 = 0.5
     *   1.0/2=0.5    -> 0.5   -> 0.5*2 = 1.0
     *   2.0/2=1.0    -> 1.0   -> 1.0*2 = 2.0
     *   3.0/2=1.5    -> 1.5   -> 1.5*2 = 3.0
     *   4.0/2=2.0    -> 2.0   -> 2.0*2 = 4.0
     *   6.0/2=3.0    -> 3.0   -> 3.0*2 = 6.0
     *   0.125/2=0.0625 -> 0   -> 0.0*2 = 0.0
     *   0.75/2=0.375 -> 0.375 -> 0.375*2 = 0.75
     *   1.5/2=0.75   -> 0.75  -> 0.75*2 = 1.5
     *   2.5/2=1.25   -> 1.25  -> 1.25*2 = 2.5
     *   8.0/2=4.0    -> 4.0   -> 4.0*2 = 8.0
     *   10.0/2=5.0   -> 5.0   -> 5.0*2 = 10.0
     *   14.0/2=7.0   -> 7.0   -> 7.0*2 = 14.0
     *   16.0/2=8.0   -> 7.5 (clamp) -> 7.5*2 = 15.0 */
    constexpr std::array<float, 16> fp6_ref_data = {
        0.f, 0.25f, 0.5f, 1.f, 2.f, 3.f, 4.f, 6.f, 0.f, 0.75f, 1.5f, 2.5f, 8.f, 10.f, 14.f, 15.f};

    // BF6 E3M2 test values with scale=2.0: range [0.0625, 28], scaled range [0.125, 56]
    constexpr std::array<float, 16> bf6_test_data = {0.f,
                                                     0.125f,
                                                     0.25f,
                                                     0.5f,
                                                     1.f,
                                                     2.f,
                                                     4.f,
                                                     6.f,
                                                     0.0625f,
                                                     0.375f,
                                                     1.5f,
                                                     3.f,
                                                     12.f,
                                                     24.f,
                                                     28.f,
                                                     32.f};
    /* Expected results after: input/scale -> BF6 quantize -> output*scale
     * For scale=2.0:
     *   0.0/2=0.0      -> 0      -> 0.0*2 = 0.0
     *   0.125/2=0.0625 -> 0.0625 -> 0.0625*2 = 0.125
     *   0.25/2=0.125   -> 0.125  -> 0.125*2 = 0.25
     *   0.5/2=0.25     -> 0.25   -> 0.25*2 = 0.5
     *   1.0/2=0.5      -> 0.5    -> 0.5*2 = 1.0
     *   2.0/2=1.0      -> 1.0    -> 1.0*2 = 2.0
     *   4.0/2=2.0      -> 2.0    -> 2.0*2 = 4.0
     *   6.0/2=3.0      -> 3.0    -> 3.0*2 = 6.0
     *   0.0625/2=0.03125 -> 0    -> 0.0*2 = 0.0
     *   0.375/2=0.1875 -> 0.1875 -> 0.1875*2 = 0.375
     *   1.5/2=0.75     -> 0.75   -> 0.75*2 = 1.5
     *   3.0/2=1.5      -> 1.5    -> 1.5*2 = 3.0
     *   12.0/2=6.0     -> 6.0    -> 6.0*2 = 12.0
     *   24.0/2=12.0    -> 12.0   -> 12.0*2 = 24.0
     *   28.0/2=14.0    -> 14.0   -> 14.0*2 = 28.0
     *   32.0/2=16.0    -> 16.0   -> 16.0*2 = 32.0 */
    constexpr std::array<float, 16> bf6_ref_data = {0.f,
                                                    0.125f,
                                                    0.25f,
                                                    0.5f,
                                                    1.f,
                                                    2.f,
                                                    4.f,
                                                    6.f,
                                                    0.f,
                                                    0.375f,
                                                    1.5f,
                                                    3.f,
                                                    12.f,
                                                    24.f,
                                                    28.f,
                                                    32.f};

    // Select test data based on PK6 type
    const auto& test_data = (std::is_same_v<PK6, pk_fp6_t> ? fp6_test_data : bf6_test_data);
    const auto& ref_data  = (std::is_same_v<PK6, pk_fp6_t> ? fp6_ref_data : bf6_ref_data);

    static_assert(fp6_test_data.size() == fp6_ref_data.size());
    static_assert(bf6_test_data.size() == bf6_ref_data.size());

    constexpr int N = 16;
    std::array<SRC, N> in;
    std::array<DST, N> ref, out;

    // Prepare input and ground truth on host
    for(int i = 0; i < N; ++i)
    {
        in[i]  = toSRC(test_data[i]);
        ref[i] = toDST(ref_data[i]);
        EXPECT_EQ(test_data[i], toF32(in[i]));
        EXPECT_EQ(ref_data[i], toF32(ref[i]));
    }

    using job = SrcPk6Dst<SRC, PK6, DST, N>;

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

/* Kernel for testing pkscale_type_convert with Packed4Scale for FP6/BF6 */
template <typename PK6, typename DST, int N, bool Block16Mod>
struct TestPk6scaleTypeConvert
{
    CK_TILE_DEVICE void operator()([[maybe_unused]] PK6 val,
                                   ck_tile::Packed4Scale_E8M0::raw_type* p_scale,
                                   DST* dst_data) const
    {
        if(dst_data == nullptr || p_scale == nullptr)
            return;

#if defined(__gfx125__)
        using DSTx16_t       = ck_tile::ext_vector_t<DST, 16>;
        ck_tile::index_t lid = __lane_id();
        ck_tile::Packed4Scale_E8M0 scale(p_scale[lid]);

        ck_tile::static_for<0, 4, 1>{}([&](auto it) {
            constexpr int opsel = (Block16Mod) ? (it + 4) : it;
            auto vT16           = ck_tile::pk6scaled_type_convert<DSTx16_t, PK6, opsel>(val, scale);

            /* Row index of dst_data:
             * (lid & 0x0F): mapping lane0-15 and 16-31 to row 0-15
             * Column index of p_mat:
             *  it * 32: each iteration process 32 columns
             * ((lid >> 4) & 1) * 16: lane 0-15 write first 16 columns
             *                        lane 16-31 write the next 16 columns*/
            ck_tile::static_for<0, 16, 1>{}([&](auto ii) {
                dst_data[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii] =
                    vT16[static_cast<int>(ii)];
            });
        });
#endif
    }
};

template <typename PK6, typename DST, bool Block16Mod>
void test_pkscale_type_convert_device()
{
    // matrix shape M x N
    constexpr int M        = 16;
    constexpr int N        = 128;
    constexpr int N_scale  = N / 16; // every 16 elements share a scale in packed-16 type convert
    constexpr int mat_fval = 1.0f;
    std::vector<DST> out(M * N);

    int scale_init_option = 0; // 0: fixed value on the same column, 1: random values

    /* From a float scale matrix [M * N_scale=8] to a packed-4 scale matrix [M * 2] */
    /*Each 16 elements share one scale factor
      n:       [0:15]     [16:31]    [32:47]    [48:63]    [64:79]    [80:95]    [96:111] [112:127]
      index:   0          1          2          3          4          5          6          7
      m[0:15]  fscale[m][0] ...
    */
    std::vector<float> fscale(M * N_scale);
    if(scale_init_option == 0)
    {
        // Option 0: Fixed pattern with wide dynamic range (same for all rows)
        // Note: Values chosen to be safe for fp16 (max ~= 65504)
        for(int m = 0; m < M; m++)
        {
            fscale[m * N_scale + 0] = std::pow(2.0f, -10.0f); // 2^-10 ~= 0.000977
            fscale[m * N_scale + 1] = std::pow(2.0f, -5.0f);  // 2^-5  = 0.03125
            fscale[m * N_scale + 2] = std::pow(2.0f, 8.0f);   // 2^8   = 256
            fscale[m * N_scale + 3] = std::pow(2.0f, 15.0f);  // 2^15  = 32768 (safe for fp16)
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

    std::vector<ck_tile::Packed4Scale_E8M0::raw_type> scale(2 * M);
    for(int m = 0; m < M; m++)
    {
        if constexpr(Block16Mod)
        {
            /* Each iteration take care of 16 x 128 matrix
             * opsel-4, use scale[th0:15]   [7:0]->col[0:15],   [23:16]->col[16:31]
             * opsel-5, use scale[th16:31]  [7:0]->col[0:15],   [23:16]->col[16:31]
             * opsel-6, use scale[th0:15]   [15:8]->col[32:47], [31:24]->col[48:63]
             * opsel-7, use scale[th16:31]  [15:8]->col[32:47], [31:24]->col[48:63] */
            ck_tile::Packed4Scale_E8M0 scale4(fscale[m * N_scale + 5],
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
        {
            // Block32Mod
            /* Each iteration take care of 16 x 128 matrix
             * opsel-0, use scale[th0:15]   [7:0]->col[0:15],   [15:8]->col[16:31]
             * opsel-1, use scale[th16:31]  [7:0]->col[0:15],   [15:8]->col[16:31]
             * opsel-2, use scale[th0:15]   [23:16]->col[32:47], [31:24]->col[48:63]
             * opsel-3, use scale[th16:31]  [23:16]->col[32:47], [31:24]->col[48:63] */
            ck_tile::Packed4Scale_E8M0 scale4(fscale[m * N_scale + 5],
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
     * pack 16 float data to pk_fp6_t or pk_bf6_t */
#if CK_TILE_AVX512F_WA
    fp32x8_t input_f32[2] = {fp32x8_t(mat_fval), fp32x8_t(mat_fval)};
    PK6 pk6_val           = ck_tile::type_convert<PK6>(input_f32);
#else
    PK6 pk6_val = ck_tile::type_convert<PK6>(fp32x16_t(mat_fval));
#endif

    ck_tile::DeviceMem device_out(M * N * sizeof(DST));
    ck_tile::DeviceMem device_scale(2 * M * sizeof(ck_tile::Packed4Scale_E8M0::raw_type));
    device_scale.ToDevice(scale.data());

    using kernel = TestPk6scaleTypeConvert<PK6, DST, N, Block16Mod>;
    MyKernel<kernel><<<1, 32>>>(
        pk6_val,
        reinterpret_cast<ck_tile::Packed4Scale_E8M0::raw_type*>(device_scale.GetDeviceBuffer()),
        reinterpret_cast<DST*>(device_out.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    // Verify results
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++)
        {
            int scale_idx  = n / 16;
            float expected = mat_fval * fscale[m * N_scale + scale_idx];
            DST out_val    = out[m * N + n];
            EXPECT_FLOAT_EQ(toF32(out_val), expected)
                << "Mismatch at [" << m << "][" << n << "]: expected " << expected << " got "
                << toF32(out_val) << " (scale=" << fscale[m * N_scale + scale_idx] << ")";
        }
    }
}
