// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only checks of mx_scale_to_kmajor (host/mx_processing.hpp) and
// host/reference/reference_mx_gemm_tdm_scheduled.hpp: the K-major scale shuffle, the element
// codecs, the epilogue's output rounding, and the K-major reference against a plain one.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/host/mx_processing.hpp"
#include "ck_tile/host/reference/reference_mx_gemm_tdm_scheduled.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::long_index_t;
namespace sched = ck_tile::mx_tdm_scheduled;

/// Bytes of a K-major scale array of `rows` rows.
std::size_t mx_scale_kmajor_bytes(index_t rows, long_index_t K)
{
    return static_cast<std::size_t>(ck_tile::integer_divide_ceil(K, sched::kScaleStepK)) * rows * 4;
}

/// The inverse of mx_scale_to_kmajor: [rows][ceil(K / 32)], dense.
std::vector<uint8_t> mx_scale_from_kmajor(const uint8_t* kmajor, index_t rows, long_index_t K)
{
    const long_index_t groups = ck_tile::integer_divide_ceil(K, sched::kScaleGroupK);
    std::vector<uint8_t> out(static_cast<std::size_t>(rows * groups));
    for(index_t r = 0; r < rows; ++r)
        for(long_index_t g = 0; g < groups; ++g)
            out[r * groups + g] = kmajor[((g / 4) * rows + r) * 4 + g % 4];
    return out;
}

TEST(MxGemmTdmScheduledHostRef, ScaleShuffleRoundTripsAndPadsThePartialKStep)
{
    std::mt19937 rng(7);
    for(long_index_t K : {128, 256, 1024, 32, 96, 160, 4128})
    {
        SCOPED_TRACE(K);
        const index_t rows        = 48;
        const long_index_t groups = (K + 31) / 32;
        const long_index_t stride = groups + 3;
        std::vector<uint8_t> natural(rows * stride);
        for(auto& x : natural)
            x = static_cast<uint8_t>(1 + rng() % 254);
        const auto kmajor = ck_tile::mx_scale_to_kmajor(natural.data(), rows, K, stride);
        ASSERT_EQ(kmajor.size(), mx_scale_kmajor_bytes(rows, K));
        ASSERT_EQ(kmajor.size(), static_cast<size_t>((K + 127) / 128 * rows * 4));
        const auto back = mx_scale_from_kmajor(kmajor.data(), rows, K);
        for(index_t r = 0; r < rows; ++r)
            for(long_index_t g = 0; g < groups; ++g)
                ASSERT_EQ(back[r * groups + g], natural[r * stride + g]);
        // Step s of row r is one int32; its byte g covers K values s * 128 + g * 32.
        const long_index_t steps = (K + 127) / 128;
        for(long_index_t s = 0; s < steps; ++s)
            for(index_t r = 0; r < rows; ++r)
                for(long_index_t g = 0; g < 4; ++g)
                {
                    const uint8_t want = s * 4 + g < groups ? natural[r * stride + s * 4 + g] : 0;
                    ASSERT_EQ(kmajor[(s * rows + r) * 4 + g], want);
                }
    }
}

TEST(MxGemmTdmScheduledHostRef, Fp8CodecsRoundTripEveryCode)
{
    for(int code = 0; code < 256; ++code)
    {
        const float v = sched::fp8_e4m3_to_float(static_cast<uint8_t>(code));
        if((code & 0x7F) == 0x7F)
        {
            EXPECT_TRUE(std::isnan(v)) << code;
            EXPECT_EQ(sched::float_to_fp8_e4m3_rne(v) & 0x7F, 0x7F);
            continue;
        }
        EXPECT_EQ(sched::float_to_fp8_e4m3_rne(v), code) << code << " " << v;
    }
    EXPECT_EQ(sched::fp8_e4m3_to_float(0x7E), 448.0f);
    EXPECT_EQ(sched::fp8_e4m3_to_float(0x01), std::ldexp(1.0f, -9));
    EXPECT_EQ(sched::bf8_e5m2_to_float(0x7B), 57344.0f);
    EXPECT_TRUE(std::isinf(sched::bf8_e5m2_to_float(0x7C)));
    EXPECT_EQ(sched::fp4_e2m1_to_float(0x7), 6.0f);
    EXPECT_EQ(sched::fp4_e2m1_to_float(0xF), -6.0f);
    EXPECT_EQ(sched::e8m0_to_float(127), 1.0f);
    EXPECT_EQ(sched::e8m0_to_float(0), std::ldexp(1.0f, -127));
    EXPECT_TRUE(std::isnan(sched::e8m0_to_float(0xFF)));
    const uint8_t row[2] = {0x21, 0xF7};
    EXPECT_EQ(sched::decode_element<ck_tile::pk_fp4_t>(row, 0), 0.5f);
    EXPECT_EQ(sched::decode_element<ck_tile::pk_fp4_t>(row, 1), 1.0f);
    EXPECT_EQ(sched::decode_element<ck_tile::pk_fp4_t>(row, 2), 6.0f);
    EXPECT_EQ(sched::decode_element<ck_tile::pk_fp4_t>(row, 3), -6.0f);
}

TEST(MxGemmTdmScheduledHostRef, Fp8RoundsToNearestEven)
{
    // Between 1.0 (0x38) and 1.125 (0x39): the midpoint goes to the even code.
    EXPECT_EQ(sched::float_to_fp8_e4m3_rne(1.0625f), 0x38);
    EXPECT_EQ(sched::float_to_fp8_e4m3_rne(1.0626f), 0x39);
    EXPECT_EQ(sched::float_to_fp8_e4m3_rne(1.1875f), 0x3A);
    // Subnormals: quantum 2^-9.
    EXPECT_EQ(sched::float_to_fp8_e4m3_rne(std::ldexp(1.5f, -9)), 0x02);
    EXPECT_EQ(sched::float_to_fp8_e4m3_rne(std::ldexp(0.5f, -9)), 0x00);
    EXPECT_EQ(sched::float_to_fp8_e4m3_rne(-std::ldexp(0.5f, -9)), 0x80);
    EXPECT_EQ(sched::float_to_fp8_e4m3_rne(std::ldexp(7.5f, -9)), 0x08);
    EXPECT_EQ(sched::float_to_fp8_e4m3_rne(448.0f), 0x7E);
}

TEST(MxGemmTdmScheduledHostRef, Fp8SaturatesAsTheEpilogue)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(sched::saturate_fp8(1000.0f), 448.0f);
    EXPECT_EQ(sched::saturate_fp8(-1e30f), -448.0f);
    EXPECT_TRUE(std::isnan(sched::saturate_fp8(inf)));
    EXPECT_TRUE(std::isnan(sched::saturate_fp8(-inf)));
    EXPECT_TRUE(std::isnan(sched::saturate_fp8(nan)));
    EXPECT_EQ(sched::saturate_fp8(3.0f), 3.0f);
    uint8_t d[4];
    sched::store_out<ck_tile::fp8_t>(d, 0, 500.0f);
    sched::store_out<ck_tile::fp8_t>(d, 1, -inf);
    sched::store_out<ck_tile::fp8_t>(d, 2, 449.0f);
    sched::store_out<ck_tile::fp8_t>(d, 3, -0.0f);
    EXPECT_EQ(d[0], 0x7E);
    EXPECT_EQ(d[1] & 0x7F, 0x7F);
    EXPECT_EQ(d[2], 0x7E);
    EXPECT_EQ(d[3], 0x80);
}

TEST(MxGemmTdmScheduledHostRef, Fp16AndBf16RoundToNearestEven)
{
    std::mt19937 rng(11);
    for(int i = 0; i < 200000; ++i)
    {
        const uint32_t bits = rng();
        const float v       = sched::bits_float(bits);
        if(std::isnan(v))
            continue;
        const _Float16 h = static_cast<_Float16>(v);
        uint16_t want;
        std::memcpy(&want, &h, 2);
        ASSERT_EQ(sched::float_to_fp16_rne(v), want) << v;
        if(!std::isinf(v))
            ASSERT_EQ(sched::fp16_bits_to_float(want), static_cast<float>(h));
    }
    EXPECT_EQ(sched::float_to_fp16_rne(65519.0f), 0x7BFF);
    EXPECT_EQ(sched::float_to_fp16_rne(65520.0f), 0x7C00);
    // bf16: 1 + 2^-8 is the midpoint of 1 and 1 + 2^-7; 1 + 3 * 2^-8 goes up to the even code.
    EXPECT_EQ(sched::float_to_bf16_rne(1.0f + std::ldexp(1.0f, -8)), 0x3F80);
    EXPECT_EQ(sched::float_to_bf16_rne(1.0f + 3 * std::ldexp(1.0f, -8)), 0x3F82);
    EXPECT_EQ(sched::float_to_bf16_rne(std::numeric_limits<float>::max()), 0x7F80);
    EXPECT_EQ(sched::float_to_bf16_rne(std::numeric_limits<float>::quiet_NaN()) & 0x7FC0, 0x7FC0);
    EXPECT_EQ(sched::bf16_bits_to_float(0x3F82), 1.0f + std::ldexp(1.0f, -6));
}

struct Operands
{
    std::vector<uint8_t> a, b, a_natural, b_natural, a_scale, b_scale;
};

// Exact-sum inputs: elements in {-2, -1, 0, 1, 2} (FP4 / FP8 codes), scales 2^-1..2^1.
template <typename T>
Operands make_operands(index_t M, index_t N, long_index_t K, uint32_t seed)
{
    std::mt19937 rng(seed);
    const long_index_t rb     = sched::row_bytes<T>(K);
    const long_index_t groups = (K + 31) / 32;
    Operands o;
    auto fill = [&](index_t rows) {
        std::vector<uint8_t> x(static_cast<std::size_t>(rows * rb), 0);
        static constexpr uint8_t kFp4[5] = {0xC, 0xA, 0x0, 0x2, 0x4};
        static constexpr uint8_t kFp8[5] = {0xC0, 0xB8, 0x00, 0x38, 0x40};
        static constexpr uint8_t kBf8[5] = {0xC0, 0xBC, 0x00, 0x3C, 0x40};
        for(index_t r = 0; r < rows; ++r)
            for(long_index_t k = 0; k < K; ++k)
            {
                const int pick = rng() % 5;
                if constexpr(std::is_same_v<T, ck_tile::pk_fp4_t>)
                    x[r * rb + k / 2] |= kFp4[pick] << (4 * (k % 2));
                else
                    x[r * rb + k] = std::is_same_v<T, ck_tile::fp8_t> ? kFp8[pick] : kBf8[pick];
            }
        return x;
    };
    o.a         = fill(M);
    o.b         = fill(N);
    auto scales = [&](index_t rows) {
        std::vector<uint8_t> s(static_cast<std::size_t>(rows * groups));
        for(auto& x : s)
            x = static_cast<uint8_t>(126 + rng() % 3);
        return s;
    };
    o.a_natural = scales(M);
    o.b_natural = scales(N);
    o.a_scale   = ck_tile::mx_scale_to_kmajor(o.a_natural.data(), M, K, groups);
    o.b_scale   = ck_tile::mx_scale_to_kmajor(o.b_natural.data(), N, K, groups);
    return o;
}

template <typename T, typename O>
void check_kmajor_reference()
{
    for(long_index_t K : {128, 160, 512})
    {
        SCOPED_TRACE(K);
        const index_t M = 37, N = 23;
        const Operands o          = make_operands<T>(M, N, K, 3 + static_cast<uint32_t>(K));
        const long_index_t rb     = sched::row_bytes<T>(K);
        const long_index_t groups = (K + 31) / 32;
        const index_t ob          = sched::out_bytes<O>();
        std::vector<uint8_t> d(M * N * ob), want(M * N * ob);
        ck_tile::reference_mx_gemm_tdm_scheduled<T, T, O>(o.a.data(),
                                                          rb,
                                                          o.b.data(),
                                                          rb,
                                                          o.a_scale.data(),
                                                          o.b_scale.data(),
                                                          d.data(),
                                                          N,
                                                          M,
                                                          N,
                                                          K,
                                                          4);
        for(index_t m = 0; m < M; ++m)
            for(index_t n = 0; n < N; ++n)
            {
                float acc = 0.0f;
                for(long_index_t k = 0; k < K; ++k)
                    acc += sched::decode_element<T>(&o.a[m * rb], k) *
                           sched::e8m0_to_float(o.a_natural[m * groups + k / 32]) *
                           sched::decode_element<T>(&o.b[n * rb], k) *
                           sched::e8m0_to_float(o.b_natural[n * groups + k / 32]);
                sched::store_out<O>(want.data(), m * N + n, acc);
            }
        ASSERT_EQ(d, want);
    }
}

template <typename T>
void check_kmajor_reference_outs()
{
    check_kmajor_reference<T, ck_tile::fp8_t>();
    check_kmajor_reference<T, ck_tile::half_t>();
    check_kmajor_reference<T, ck_tile::bf16_t>();
}

TEST(MxGemmTdmScheduledHostRef, KMajorReferenceEqualsThePlainReference)
{
    check_kmajor_reference_outs<ck_tile::pk_fp4_t>();
    check_kmajor_reference_outs<ck_tile::fp8_t>();
    check_kmajor_reference_outs<ck_tile::bf8_t>();
}

// A row subset is those rows of the full E, stored compactly.
TEST(MxGemmTdmScheduledHostRef, RowSubsetEqualsTheFullRows)
{
    const index_t M = 41, N = 19;
    const long_index_t K = 256;
    const Operands o     = make_operands<ck_tile::fp8_t>(M, N, K, 13);
    std::vector<uint16_t> full(M * N);
    ck_tile::reference_mx_gemm_tdm_scheduled<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bf16_t>(
        o.a.data(),
        K,
        o.b.data(),
        K,
        o.a_scale.data(),
        o.b_scale.data(),
        full.data(),
        N,
        M,
        N,
        K,
        3);
    const std::vector<index_t> rows{0, 7, 8, 40};
    std::vector<uint16_t> part(rows.size() * N);
    ck_tile::reference_mx_gemm_tdm_scheduled<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bf16_t>(
        o.a.data(),
        K,
        o.b.data(),
        K,
        o.a_scale.data(),
        o.b_scale.data(),
        part.data(),
        N,
        M,
        N,
        K,
        2,
        rows.data(),
        static_cast<index_t>(rows.size()));
    for(std::size_t i = 0; i < rows.size(); ++i)
        for(index_t n = 0; n < N; ++n)
            EXPECT_EQ(part[i * N + n], full[rows[i] * N + n]) << "row " << rows[i] << " col " << n;
}

} // namespace
