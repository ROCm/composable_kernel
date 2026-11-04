// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Host-only tests of the MX GEMM test helpers in test_mx_gemm_host_helpers.hpp and of the
 *        host reference reference_mx_gemm. Nothing is launched.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>

#include "gtest/gtest.h"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "test_mx_gemm_host_helpers.hpp"

namespace {

using ck_tile::index_t;
using F6  = ck_tile::pk_fp6x16_t;
using BF6 = ck_tile::pk_bf6x16_t;
using F4  = ck_tile::pk_fp4_t;
using F8  = ck_tile::fp8_t;

// ------------------------------------------------------------------------------------------
// 6-bit packs
// ------------------------------------------------------------------------------------------

// Reference decoders written from the OCP formats, independent of the CK conversion code:
// sign bit 5, then the exponent and mantissa fields; subnormal when the exponent field is 0.
float decode_6bit(int code, int mant_bits, int bias)
{
    const int sign     = (code >> 5) & 1;
    const int exp_bits = 5 - mant_bits;
    const int exponent = (code >> mant_bits) & ((1 << exp_bits) - 1);
    const int mantissa = code & ((1 << mant_bits) - 1);
    const float frac   = static_cast<float>(mantissa) / static_cast<float>(1 << mant_bits);
    const float value =
        exponent == 0 ? std::ldexp(frac, 1 - bias) : std::ldexp(1.0f + frac, exponent - bias);
    return sign ? -value : value;
}
float decode_e2m3(int code) { return decode_6bit(code, 3, 1); }
float decode_e3m2(int code) { return decode_6bit(code, 2, 3); }

template <typename T>
void check_every_code_in_every_position(float (*decode)(int))
{
    for(int code = 0; code < 64; ++code)
    {
        for(index_t pos = 0; pos < 16; ++pos)
        {
            T pack{};
            pack.pack(code, pos);
            for(index_t i = 0; i < 16; ++i)
            {
                EXPECT_EQ(pack.unpack(i), i == pos ? code : 0) << code << " " << pos << " " << i;
            }
            EXPECT_EQ(mx_test::pk_f6x16_to_float(pack, pos), decode(code)) << code << " " << pos;
        }
    }
}

TEST(MxGemmHostHelpers, Fp6EveryCodeInEveryPosition)
{
    check_every_code_in_every_position<F6>(decode_e2m3);
}

TEST(MxGemmHostHelpers, Bf6EveryCodeInEveryPosition)
{
    check_every_code_in_every_position<BF6>(decode_e3m2);
}

TEST(MxGemmHostHelpers, Fp6DecodeMatchesTheReferenceDecoder)
{
    // reference_mx_gemm decodes pk_fp6x16_t with the legacy E2M3 decoder.
    for(int code = 0; code < 64; ++code)
    {
        EXPECT_EQ(F6::fp6_e2m3_to_float(code), decode_e2m3(code)) << code;
    }
}

TEST(MxGemmHostHelpers, SixBitDecodeKnownValues)
{
    EXPECT_EQ(decode_e2m3(0b001000), 1.0f);
    EXPECT_EQ(decode_e2m3(0b011111), 7.5f);
    EXPECT_EQ(decode_e2m3(0b000001), 0.125f);
    EXPECT_EQ(decode_e2m3(0b101000), -1.0f);
    EXPECT_EQ(decode_e3m2(0b001100), 1.0f);
    EXPECT_EQ(decode_e3m2(0b011111), 28.0f);
    EXPECT_EQ(decode_e3m2(0b000001), 0.0625f);
    EXPECT_EQ(decode_e3m2(0b101100), -1.0f);
}

TEST(MxGemmHostHelpers, IsPkF6)
{
    EXPECT_TRUE(ck_tile::is_pk_f6_v<F6>);
    EXPECT_TRUE(ck_tile::is_pk_f6_v<BF6>);
    EXPECT_TRUE(ck_tile::is_pk_f6_v<ck_tile::pk_fp6x32_t>);
    EXPECT_TRUE(ck_tile::is_pk_f6_v<ck_tile::pk_bf6x32_t>);
    EXPECT_TRUE(ck_tile::is_pk_f6_v<ck_tile::pk_fp6_t>);
    EXPECT_TRUE(ck_tile::is_pk_f6_v<ck_tile::pk_bf6_t>);
    EXPECT_TRUE(ck_tile::is_pk_f6_v<const F6&>);
    EXPECT_FALSE(ck_tile::is_pk_f6_v<F4>);
    EXPECT_FALSE(ck_tile::is_pk_f6_v<ck_tile::fp8_t>);
    EXPECT_FALSE(ck_tile::is_pk_f6_v<ck_tile::bf16_t>);
}

template <typename T>
void check_proxy_fill(std::size_t rows, std::size_t cols)
{
    ck_tile::HostTensor<T> tensor({rows, cols}, {cols, std::size_t{1}});
    // Poison every pack: all-ones codes decode to negative values, outside the fill range.
    for(auto& pack : tensor.mData)
    {
        std::memset(&pack, 0xFF, sizeof(pack));
    }
    mx_test::fill_mx_operand(tensor, 0.0f, 4.0f, 1234);

    std::set<float> seen;
    for(std::size_t r = 0; r < rows; ++r)
    {
        for(std::size_t c = 0; c < cols; ++c)
        {
            const float v = mx_test::mx_operand_value(tensor, r, c);
            EXPECT_GE(v, 0.0f) << r << " " << c;
            EXPECT_LE(v, 4.0f) << r << " " << c;
            seen.insert(v);
        }
    }
    EXPECT_GE(seen.size(), std::min<std::size_t>(rows * cols, 8));

    // The legacy pack holds the proxy's first 12 bytes, in the same element order.
    using Proxy = mx_test::mx_init_proxy_t<T>;
    ck_tile::HostTensor<Proxy> proxy(tensor.mDesc);
    ck_tile::FillUniformDistribution<Proxy>{0.0f, 4.0f, 1234}(proxy);
    for(std::size_t p = 0; p < tensor.mData.size(); ++p)
    {
        for(index_t i = 0; i < 16; ++i)
        {
            EXPECT_EQ(tensor.mData[p].unpack(i), proxy.mData[p].get_element(i)) << p << " " << i;
        }
    }
}

TEST(MxGemmHostHelpers, Fp6FillWritesEveryElement)
{
    // 1, 5, 17 and 33 packs: none is a multiple of the proxy filler's 4-pack (64-byte) block, so
    // a filler that dropped a partial last block would leave poisoned packs.
    check_proxy_fill<F6>(1, 16);
    check_proxy_fill<F6>(5, 16);
    check_proxy_fill<F6>(1, 272);
    check_proxy_fill<F6>(3, 176);
}

TEST(MxGemmHostHelpers, Bf6FillWritesEveryElement)
{
    check_proxy_fill<BF6>(1, 16);
    check_proxy_fill<BF6>(5, 16);
    check_proxy_fill<BF6>(1, 272);
    check_proxy_fill<BF6>(3, 176);
}

TEST(MxGemmHostHelpers, Fp4OperandValueFollowsTheDescriptorOffset)
{
    // Element (r, c) of a row-major (2, 4) FP4 tensor is nibble (r * 4 + c) % 2 of pack
    // (r * 4 + c) / 2.
    ck_tile::HostTensor<F4> tensor({std::size_t{2}, std::size_t{4}},
                                   {std::size_t{4}, std::size_t{1}});
    const float values[8] = {0.5f, 1.0f, 1.5f, 2.0f, -3.0f, 4.0f, -6.0f, 0.0f};
    for(std::size_t p = 0; p < 4; ++p)
    {
        tensor.mData[p] = F4::_pack(ck_tile::float_to_mxfp4(values[2 * p], 1.f),
                                    ck_tile::float_to_mxfp4(values[2 * p + 1], 1.f));
    }
    for(std::size_t r = 0; r < 2; ++r)
    {
        for(std::size_t c = 0; c < 4; ++c)
        {
            EXPECT_EQ(mx_test::mx_operand_value(tensor, r, c), values[r * 4 + c]) << r << " " << c;
        }
    }
}

// ------------------------------------------------------------------------------------------
// E8M0 scales
// ------------------------------------------------------------------------------------------

TEST(MxGemmHostHelpers, RandomE8M0ScalesAreInRangeAndVary)
{
    constexpr std::size_t rows = 64, blocks = 16;
    ck_tile::HostTensor<ck_tile::e8m0_t> scales({rows, blocks}, {blocks, std::size_t{1}});
    mx_test::fill_random_e8m0(scales, -4, 4, 42);

    std::set<int> codes;
    std::size_t adjacent_differ = 0;
    for(std::size_t r = 0; r < rows; ++r)
    {
        for(std::size_t b = 0; b < blocks; ++b)
        {
            const int code = scales(r, b).get();
            EXPECT_GE(code, 127 - 4);
            EXPECT_LE(code, 127 + 4);
            codes.insert(code);
            if(b > 0 && scales(r, b) != scales(r, b - 1))
            {
                ++adjacent_differ;
            }
        }
    }
    EXPECT_EQ(codes.size(), 9u);
    // Uniform over 9 values: neighbouring blocks differ 8 times in 9.
    EXPECT_GT(adjacent_differ, rows * (blocks - 1) / 2);
}

TEST(MxGemmHostHelpers, RandomE8M0ScalesAvoidZeroAndNaN)
{
    ck_tile::HostTensor<ck_tile::e8m0_t> scales({std::size_t{4096}});
    mx_test::fill_random_e8m0(scales, -126, 127, 7);
    for(const auto& s : scales.mData)
    {
        EXPECT_NE(s.get(), 0);
        EXPECT_NE(s.get(), 0xFF);
    }
    EXPECT_THROW(mx_test::fill_random_e8m0(scales, -127, 0, 7), std::invalid_argument);
    EXPECT_THROW(mx_test::fill_random_e8m0(scales, 2, 1, 7), std::invalid_argument);
}

TEST(MxGemmHostHelpers, E8M0ExponentRangeBoundsTheOutput)
{
    for(index_t K : {128, 512, 6144})
    {
        for(float c_max : {65504.0f, 3.38953139e38f})
        {
            const auto [lo, hi] = mx_test::e8m0_exponent_range(K, 7.5f, 7.5f, c_max);
            const double bound  = static_cast<double>(K) * 7.5 * 7.5;
            EXPECT_LE(bound * std::ldexp(1.0, 2 * hi), 0.5 * c_max) << K;
            EXPECT_GT(bound * std::ldexp(1.0, 2 * (hi + 1)), 0.5 * c_max) << K;
            EXPECT_EQ(lo, hi - 4);
        }
    }
    // The fp16 gtests: K = 512 and integer data in [-5, 5] allow scales up to 2^0.
    EXPECT_EQ(mx_test::e8m0_exponent_range(512, 5.0f, 5.0f, 65504.0f).second, 0);
    // A bound too large for any scale clamps hi up to -126 + span; the range then cannot keep the
    // output below c_max, as documented.
    EXPECT_EQ(mx_test::e8m0_exponent_range(1, 1e30f, 1e30f, 1e-30f).first, -126);
    EXPECT_THROW(mx_test::e8m0_exponent_range(0, 1.0f, 1.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(mx_test::e8m0_exponent_range(128, 0.0f, 1.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(mx_test::e8m0_exponent_range(128, 1.0f, 1.0f, 1.0f, -1), std::invalid_argument);
}

// ------------------------------------------------------------------------------------------
// Tolerance
// ------------------------------------------------------------------------------------------

TEST(MxGemmHostHelpers, MantissaBitsMatchTheFormats)
{
    EXPECT_EQ(mx_test::mx_mantissa_bits<F6>, 3);  // E2M3
    EXPECT_EQ(mx_test::mx_mantissa_bits<BF6>, 2); // E3M2
    EXPECT_EQ(mx_test::mx_mantissa_bits<ck_tile::fp8_t>, 3);
    EXPECT_EQ(mx_test::mx_mantissa_bits<ck_tile::bf8_t>, 2);
    EXPECT_EQ(mx_test::mx_mantissa_bits<F4>, 1);
    EXPECT_EQ(mx_test::mx_mantissa_bits<ck_tile::half_t>, 10);
    EXPECT_EQ(mx_test::mx_mantissa_bits<ck_tile::bf16_t>, 7);
}

TEST(MxGemmHostHelpers, ToleranceFormula)
{
    // |out - ref| <= atol + rtol * |ref| with
    //   rtol = 2^-mant(C)                       one ULP of C: device and host each round once
    //   atol = 2 * K * 2^-24 * S_max            fp32 accumulation of K products, two orders
    //        + (2^-(mant(A)+1) + 2^-(mant(B)+1)) * S_max   only against an unquantized reference
    constexpr index_t K = 512;
    constexpr double S  = 1000.0;
    {
        const auto t = mx_test::mx_gemm_tolerance<F6, F6, ck_tile::bf16_t>(K, S);
        EXPECT_DOUBLE_EQ(t.rtol, 1.0 / 128);
        EXPECT_DOUBLE_EQ(t.atol, 2.0 * K * S / 16777216.0);
    }
    {
        const auto t = mx_test::mx_gemm_tolerance<F6, F6, ck_tile::half_t>(K, S);
        EXPECT_DOUBLE_EQ(t.rtol, 1.0 / 1024);
        EXPECT_DOUBLE_EQ(t.atol, 2.0 * K * S / 16777216.0);
    }
    {
        const auto t = mx_test::mx_gemm_tolerance<F6, BF6, ck_tile::bf16_t>(K, S, false);
        EXPECT_DOUBLE_EQ(t.atol, 2.0 * K * S / 16777216.0 + (1.0 / 16 + 1.0 / 8) * S);
    }
}

// mx_sum_abs_max against a direct loop over (m, n, k). M != N catches a swap of the two, random
// per-block scales catch a misindexed block, rows longer than K catch an operand read without its
// stride, and a middle row with the largest scales catches a reduction that keeps the first or the
// last row. Every product and partial sum is exact in double, so the two must agree exactly.
template <typename T>
void check_sum_abs_max(index_t M, index_t N, index_t K, index_t lda, index_t ldb)
{
    constexpr index_t kScaleBlock = 32;
    const index_t KB              = K / kScaleBlock;
    constexpr auto row            = ck_tile::bool_constant<true>{};
    constexpr auto col            = ck_tile::bool_constant<false>{};

    ck_tile::HostTensor<T> a(ck_tile::host_tensor_descriptor(M, K, lda, row));
    ck_tile::HostTensor<T> b(ck_tile::host_tensor_descriptor(K, N, ldb, col));
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_a(ck_tile::host_tensor_descriptor(M, KB, KB, row));
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_b(ck_tile::host_tensor_descriptor(KB, N, KB, col));
    mx_test::fill_mx_operand(a, -4.0f, 4.0f, 21);
    mx_test::fill_mx_operand(b, -4.0f, 4.0f, 22);
    mx_test::fill_random_e8m0(scale_a, -3, 3, 23);
    mx_test::fill_random_e8m0(scale_b, -3, 3, 24);
    // Above every random scale (at most 8), and different in adjacent blocks.
    for(index_t kb = 0; kb < KB; ++kb)
    {
        scale_a(M / 2, kb) = ck_tile::e8m0_t{kb % 2 == 0 ? 16.0f : 32.0f};
    }

    double expected  = 0.0;
    index_t best_row = -1;
    for(index_t m = 0; m < M; ++m)
    {
        for(index_t n = 0; n < N; ++n)
        {
            double s = 0.0;
            for(index_t k = 0; k < K; ++k)
            {
                s += std::abs(static_cast<double>(mx_test::mx_operand_value(a, m, k)) *
                              mx_test::mx_operand_value(b, k, n) *
                              ck_tile::type_convert<float>(scale_a(m, k / kScaleBlock)) *
                              ck_tile::type_convert<float>(scale_b(k / kScaleBlock, n)));
            }
            if(s > expected)
            {
                expected = s;
                best_row = m;
            }
        }
    }
    ASSERT_GT(expected, 0.0);
    if(M > 2)
    {
        EXPECT_EQ(best_row, M / 2);
    }
    EXPECT_EQ(mx_test::mx_sum_abs_max(a, b, scale_a, scale_b, kScaleBlock), expected)
        << M << "x" << N << "x" << K << " lda " << lda << " ldb " << ldb;
}

TEST(MxGemmHostHelpers, SumAbsMaxMatchesADirectLoop)
{
    check_sum_abs_max<F6>(1, 7, 96, 96, 96);   // one row
    check_sum_abs_max<F6>(5, 7, 96, 96, 96);   // M != N
    check_sum_abs_max<F6>(5, 7, 96, 112, 144); // rows longer than K, by one and three packs
    // FP4 shares the helper through the bf16 C tests.
    check_sum_abs_max<F4>(6, 4, 64, 64, 64);
}

// mx_sum_abs_max rejects a non-positive scale block size, and shapes it would otherwise read out
// of bounds since HostTensor does not check bounds.
TEST(MxGemmHostHelpers, SumAbsMaxRejectsInvalidArguments)
{
    constexpr index_t kScaleBlock = 32;
    constexpr index_t M = 4, N = 4, K = 64, KB = K / kScaleBlock;
    constexpr auto row = ck_tile::bool_constant<true>{};
    constexpr auto col = ck_tile::bool_constant<false>{};
    const ck_tile::HostTensor<F6> a(ck_tile::host_tensor_descriptor(M, K, K, row));
    const ck_tile::HostTensor<F6> b(ck_tile::host_tensor_descriptor(K, N, K, col));
    const ck_tile::HostTensor<F6> b_short(
        ck_tile::host_tensor_descriptor(K - kScaleBlock, N, K - kScaleBlock, col));
    const ck_tile::HostTensor<ck_tile::e8m0_t> scale_a(
        ck_tile::host_tensor_descriptor(M, KB, KB, row));
    const ck_tile::HostTensor<ck_tile::e8m0_t> scale_a_short(
        ck_tile::host_tensor_descriptor(M, KB - 1, KB - 1, row));
    const ck_tile::HostTensor<ck_tile::e8m0_t> scale_b(
        ck_tile::host_tensor_descriptor(KB, N, KB, col));
    const ck_tile::HostTensor<ck_tile::e8m0_t> scale_b_short(
        ck_tile::host_tensor_descriptor(KB - 1, N, KB - 1, col));

    EXPECT_THROW(mx_test::mx_sum_abs_max(a, b_short, scale_a, scale_b, kScaleBlock),
                 std::invalid_argument);
    EXPECT_THROW(mx_test::mx_sum_abs_max(a, b, scale_a_short, scale_b, kScaleBlock),
                 std::invalid_argument);
    EXPECT_THROW(mx_test::mx_sum_abs_max(a, b, scale_a, scale_b_short, kScaleBlock),
                 std::invalid_argument);
    EXPECT_THROW(mx_test::mx_sum_abs_max(a, b, scale_a, scale_b, 0), std::invalid_argument);
}

// A host-only MX GEMM: reference_mx_gemm on proxy-filled (FP6) or uniformly filled (FP4) data
// against a dequantize-then-multiply reference in double, within the derived tolerance.
template <typename T, typename CDataType>
void check_reference(index_t M, index_t N, index_t K)
{
    constexpr index_t kScaleBlock = 32;
    const index_t KB              = K / kScaleBlock;
    constexpr auto row            = ck_tile::bool_constant<true>{};
    constexpr auto col            = ck_tile::bool_constant<false>{};

    ck_tile::HostTensor<T> a(ck_tile::host_tensor_descriptor(M, K, K, row));
    ck_tile::HostTensor<T> b(ck_tile::host_tensor_descriptor(K, N, K, col));
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_a(ck_tile::host_tensor_descriptor(M, KB, KB, row));
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_b(ck_tile::host_tensor_descriptor(KB, N, KB, col));
    mx_test::fill_mx_operand(a, -4.0f, 4.0f, 11);
    mx_test::fill_mx_operand(b, -4.0f, 4.0f, 12);
    mx_test::fill_random_e8m0(scale_a, -2, 2, 13);
    mx_test::fill_random_e8m0(scale_b, -2, 2, 14);

    ck_tile::HostTensor<CDataType> c(ck_tile::host_tensor_descriptor(M, N, N, row));
    ck_tile::reference_mx_gemm<T, T, ck_tile::e8m0_t, ck_tile::e8m0_t, float, CDataType>(
        a, b, c, scale_a, scale_b);

    const double s_max = mx_test::mx_sum_abs_max(a, b, scale_a, scale_b, kScaleBlock);
    const auto tol     = mx_test::mx_gemm_tolerance<T, T, CDataType>(K, s_max);
    for(index_t m = 0; m < M; ++m)
    {
        for(index_t n = 0; n < N; ++n)
        {
            double exact = 0.0;
            for(index_t k = 0; k < K; ++k)
            {
                exact += static_cast<double>(mx_test::mx_operand_value(a, m, k)) *
                         mx_test::mx_operand_value(b, k, n) *
                         ck_tile::type_convert<float>(scale_a(m, k / kScaleBlock)) *
                         ck_tile::type_convert<float>(scale_b(k / kScaleBlock, n));
            }
            const double got = ck_tile::type_convert<float>(c(m, n));
            EXPECT_LE(std::abs(got - exact), tol.atol + tol.rtol * std::abs(exact))
                << m << " " << n << " exact " << exact << " got " << got;
        }
    }
}

TEST(MxGemmHostHelpers, Fp6ReferenceMatchesDequantizedGemmFloatC)
{
    check_reference<F6, float>(4, 4, 128);
}

TEST(MxGemmHostHelpers, Fp6ReferenceMatchesDequantizedGemmBf16C)
{
    check_reference<F6, ck_tile::bf16_t>(4, 4, 512);
}

TEST(MxGemmHostHelpers, Fp6ReferenceMatchesDequantizedGemmFp16C)
{
    check_reference<F6, ck_tile::half_t>(4, 4, 512);
}

TEST(MxGemmHostHelpers, Fp4ReferenceMatchesDequantizedGemmBf16C)
{
    check_reference<F4, ck_tile::bf16_t>(4, 4, 512);
}

// ------------------------------------------------------------------------------------------
// reference_mx_gemm with an explicit scale block size
// ------------------------------------------------------------------------------------------

constexpr index_t kExplicitScaleBlock = 32;
// Scale columns of A and rows of B in the test problems: ceil(K / 32) rounded up to a multiple of 4
// (four E8M0 scales per packed int32), which is 4 for every K up to 128.
// ReferenceChecksScaleBlockArguments uses K = 136, which needs 5, with 4 on purpose.
constexpr index_t kPaddedScaleBlocks = 4;
constexpr auto kRowMajor             = ck_tile::bool_constant<true>{};
constexpr auto kColMajor             = ck_tile::bool_constant<false>{};

// MX operands and scales with kPaddedScaleBlocks scale blocks along K, filled at random. The
// blocks past the first ceil(K / 32) are padding and hold E8M0 0xFF (NaN): a reference that read
// them would return NaN.
template <typename T>
struct PaddedScaleProblem
{
    static constexpr index_t M = 4;
    static constexpr index_t N = 4;

    explicit PaddedScaleProblem(index_t K)
        : a(ck_tile::host_tensor_descriptor(M, K, K, kRowMajor)),
          b(ck_tile::host_tensor_descriptor(K, N, K, kColMajor)),
          scale_a(ck_tile::host_tensor_descriptor(
              M, kPaddedScaleBlocks, kPaddedScaleBlocks, kRowMajor)),
          scale_b(
              ck_tile::host_tensor_descriptor(kPaddedScaleBlocks, N, kPaddedScaleBlocks, kColMajor))
    {
        mx_test::fill_mx_operand(a, -4.0f, 4.0f, 21);
        mx_test::fill_mx_operand(b, -4.0f, 4.0f, 22);
        mx_test::fill_random_e8m0(scale_a, -2, 2, 23);
        mx_test::fill_random_e8m0(scale_b, -2, 2, 24);
        const auto nan = ck_tile::numeric<ck_tile::e8m0_t>::quiet_NaN();
        for(index_t kb = ck_tile::integer_divide_ceil(K, kExplicitScaleBlock);
            kb < kPaddedScaleBlocks;
            ++kb)
        {
            for(index_t m = 0; m < M; ++m)
            {
                scale_a(m, kb) = nan;
            }
            for(index_t n = 0; n < N; ++n)
            {
                scale_b(kb, n) = nan;
            }
        }
    }

    ck_tile::HostTensor<T> a;
    ck_tile::HostTensor<T> b;
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_a;
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_b;
};

// reference_mx_gemm with an explicit 32-element scale block against a dequantize-then-multiply
// reference in double: a partial last block scales only the last K % 32 elements, and the padding
// blocks are never read.
template <typename T>
void check_explicit_scale_block(index_t K)
{
    const PaddedScaleProblem<T> p(K);
    ck_tile::HostTensor<float> c(ck_tile::host_tensor_descriptor(p.M, p.N, p.N, kRowMajor));
    ck_tile::reference_mx_gemm<T, T, ck_tile::e8m0_t, ck_tile::e8m0_t, float, float>(
        p.a, p.b, c, p.scale_a, p.scale_b, kExplicitScaleBlock);

    const double s_max =
        mx_test::mx_sum_abs_max(p.a, p.b, p.scale_a, p.scale_b, kExplicitScaleBlock);
    const auto tol = mx_test::mx_gemm_tolerance<T, T, float>(K, s_max);
    for(index_t m = 0; m < p.M; ++m)
    {
        for(index_t n = 0; n < p.N; ++n)
        {
            double exact = 0.0;
            for(index_t k = 0; k < K; ++k)
            {
                exact += static_cast<double>(mx_test::mx_operand_value(p.a, m, k)) *
                         mx_test::mx_operand_value(p.b, k, n) *
                         ck_tile::type_convert<float>(p.scale_a(m, k / kExplicitScaleBlock)) *
                         ck_tile::type_convert<float>(p.scale_b(k / kExplicitScaleBlock, n));
            }
            const double got = c(m, n);
            ASSERT_FALSE(std::isnan(got)) << "K " << K << ", " << m << " " << n;
            EXPECT_LE(std::abs(got - exact), tol.atol + tol.rtol * std::abs(exact))
                << "K " << K << ", " << m << " " << n << " exact " << exact << " got " << got;
        }
    }
}

TEST(MxGemmHostHelpers, ReferenceExplicitScaleBlockPartialLastBlock)
{
    for(const index_t K : {80, 24})
    {
        // K = 80: 3 scale blocks, the last covering 16 elements, and 1 padding block.
        // K = 24: 1 partial block of 24 elements and 3 padding blocks.
        {
            SCOPED_TRACE("F8");
            check_explicit_scale_block<F8>(K);
        }
        {
            SCOPED_TRACE("F4");
            check_explicit_scale_block<F4>(K);
        }
    }
}

TEST(MxGemmHostHelpers, ReferenceExplicitScaleBlockWholeBlocksSkipsPadding)
{
    // K = 96: 3 whole scale blocks and 1 padding block, which must not be read.
    {
        SCOPED_TRACE("F8");
        check_explicit_scale_block<F8>(96);
    }
    {
        SCOPED_TRACE("F4");
        check_explicit_scale_block<F4>(96);
    }
}

TEST(MxGemmHostHelpers, ReferenceDerivedScaleBlockMatchesExplicit)
{
    // Without padding, the derived block size (K / scale columns = 128 / 4) is the explicit one.
    const PaddedScaleProblem<F8> p(kExplicitScaleBlock * kPaddedScaleBlocks);
    ck_tile::HostTensor<float> c_derived(ck_tile::host_tensor_descriptor(p.M, p.N, p.N, kRowMajor));
    ck_tile::HostTensor<float> c_explicit(c_derived.mDesc);
    ck_tile::reference_mx_gemm<F8, F8, ck_tile::e8m0_t, ck_tile::e8m0_t, float, float>(
        p.a, p.b, c_derived, p.scale_a, p.scale_b);
    ck_tile::reference_mx_gemm<F8, F8, ck_tile::e8m0_t, ck_tile::e8m0_t, float, float>(
        p.a, p.b, c_explicit, p.scale_a, p.scale_b, kExplicitScaleBlock);
    EXPECT_EQ(c_derived.mData, c_explicit.mData);
}

TEST(MxGemmHostHelpers, ReferenceChecksScaleBlockArguments)
{
    const auto run = [](const PaddedScaleProblem<F8>& q,
                        const ck_tile::HostTensor<ck_tile::e8m0_t>& scale_a,
                        const ck_tile::HostTensor<ck_tile::e8m0_t>& scale_b,
                        index_t block) {
        ck_tile::HostTensor<float> c(ck_tile::host_tensor_descriptor(q.M, q.N, q.N, kRowMajor));
        ck_tile::reference_mx_gemm<F8, F8, ck_tile::e8m0_t, ck_tile::e8m0_t, float, float>(
            q.a, q.b, c, scale_a, scale_b, block);
    };
    // K = 136 needs ceil(136 / 32) = 5 scale blocks of 32 elements, one more than K / 32. The
    // problem has 4 for both operands; the 5-block tensors below lengthen one operand at a time,
    // so that each operand's check has to throw on its own.
    const PaddedScaleProblem<F8> p(136);
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_a5(
        ck_tile::host_tensor_descriptor(p.M, 5, 5, kRowMajor));
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_b5(
        ck_tile::host_tensor_descriptor(5, p.N, 5, kColMajor));
    mx_test::fill_random_e8m0(scale_a5, -2, 2, 25);
    mx_test::fill_random_e8m0(scale_b5, -2, 2, 26);
    EXPECT_THROW(run(p, p.scale_a, scale_b5, kExplicitScaleBlock), std::invalid_argument);
    EXPECT_THROW(run(p, scale_a5, p.scale_b, kExplicitScaleBlock), std::invalid_argument);
    EXPECT_THROW(run(p, p.scale_a, p.scale_b, -1), std::invalid_argument); // a negative block size
    EXPECT_NO_THROW(run(p, scale_a5, scale_b5, kExplicitScaleBlock)); // exactly ceil(K / 32) blocks
    EXPECT_NO_THROW(run(p, p.scale_a, p.scale_b, 64)); // reads ceil(136 / 64) = 3 of the 4 blocks

    // Scales too small along M or N.
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_a_short_m(
        ck_tile::host_tensor_descriptor(p.M - 1, 5, 5, kRowMajor));
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_b_short_n(
        ck_tile::host_tensor_descriptor(5, p.N - 1, 5, kColMajor));
    EXPECT_THROW(run(p, scale_a_short_m, scale_b5, kExplicitScaleBlock), std::invalid_argument);
    EXPECT_THROW(run(p, scale_a5, scale_b_short_n, kExplicitScaleBlock), std::invalid_argument);

    // The derived block size (0) needs K to be a multiple of the scale columns of A.
    const PaddedScaleProblem<F8> p_k2(2); // 2 / 4 would be 0
    EXPECT_THROW(run(p_k2, p_k2.scale_a, p_k2.scale_b, 0), std::invalid_argument);
    EXPECT_NO_THROW(run(p_k2, p_k2.scale_a, p_k2.scale_b, kExplicitScaleBlock));
    const PaddedScaleProblem<F8> p_k130(130); // 130 / 4 = 32.5: the last column would be overrun
    EXPECT_THROW(run(p_k130, p_k130.scale_a, p_k130.scale_b, 0), std::invalid_argument);
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_a_no_cols(
        ck_tile::host_tensor_descriptor(p.M, 0, 1, kRowMajor));
    EXPECT_THROW(run(p, scale_a_no_cols, p.scale_b, 0), std::invalid_argument);
    // Edges that must not throw: K = 0 reads no scale, and K equal to the 4 scale columns derives a
    // block size of 1.
    const PaddedScaleProblem<F8> p_k0(0);
    EXPECT_NO_THROW(run(p_k0, p_k0.scale_a, p_k0.scale_b, 0));
    const PaddedScaleProblem<F8> p_k4(kPaddedScaleBlocks);
    EXPECT_NO_THROW(run(p_k4, p_k4.scale_a, p_k4.scale_b, 0));
}

} // namespace
