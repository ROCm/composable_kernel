// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/mx_processing.hpp"

/**
 * @file
 * Reference of MxGemmPipelineAgBgCrCompTDMScheduled: bit-exact element codecs, the output
 * rounding of its epilogue and a reference GEMM over its K-major scales (mx_scale_to_kmajor in
 * host/mx_processing.hpp).
 *
 * Elements. A and B rows are K-contiguous bytes: FP4 two per byte (K even in the low nibble), FP8
 * E4M3 (OCP, no infinity) and BF8 E5M2 one per byte. C and D are row-major in the output type.
 * The decoders follow the hardware encodings, including E8M0 0x00 (2^-127) and BF8 infinities,
 * which type_convert decodes differently.
 */

namespace ck_tile {

namespace mx_tdm_scheduled {

CK_TILE_HOST float e8m0_to_float(uint8_t e)
{
    if(e == 0xFF)
        return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(e) - 127);
}

CK_TILE_HOST float fp4_e2m1_to_float(uint8_t nibble)
{
    static constexpr float kValues[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v                     = kValues[nibble & 7];
    return (nibble & 8) ? -v : v;
}

CK_TILE_HOST float fp8_e4m3_to_float(uint8_t b)
{
    const int exp  = (b >> 3) & 0xF;
    const int mant = b & 7;
    if(exp == 0xF && mant == 7)
        return std::numeric_limits<float>::quiet_NaN();
    const float v = exp == 0 ? std::ldexp(static_cast<float>(mant), -9)
                             : std::ldexp(1.0f + mant / 8.0f, exp - 7);
    return (b & 0x80) ? -v : v;
}

CK_TILE_HOST float bf8_e5m2_to_float(uint8_t b)
{
    const int exp  = (b >> 2) & 0x1F;
    const int mant = b & 3;
    float v        = 0.0f;
    if(exp == 0x1F)
        v = mant ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else
        v = exp == 0 ? std::ldexp(static_cast<float>(mant), -16)
                     : std::ldexp(1.0f + mant / 4.0f, exp - 15);
    return (b & 0x80) ? -v : v;
}

/// Element k of a K-contiguous A or B row of T (pk_fp4_t, fp8_t or bf8_t).
template <typename T>
CK_TILE_HOST float decode_element(const uint8_t* row, long_index_t k)
{
    if constexpr(std::is_same_v<T, pk_fp4_t>)
        return fp4_e2m1_to_float((row[k / 2] >> (4 * (k % 2))) & 0xF);
    else if constexpr(std::is_same_v<T, fp8_t>)
        return fp8_e4m3_to_float(row[k]);
    else
    {
        static_assert(std::is_same_v<T, bf8_t>, "decode_element: A and B are fp4, fp8 or bf8");
        return bf8_e5m2_to_float(row[k]);
    }
}

/// Bytes of an A or B row of K elements of T.
template <typename T>
CK_TILE_HOST long_index_t row_bytes(long_index_t K)
{
    return std::is_same_v<T, pk_fp4_t> ? (K + 1) / 2 : K;
}

CK_TILE_HOST uint32_t float_bits(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}

CK_TILE_HOST float bits_float(uint32_t u)
{
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

/// The epilogue's FP8 saturation: +-inf to NaN, NaN kept, finite values clamped to +-448
/// (fma(v, 0, v) then llvm.maximum / llvm.minimum).
CK_TILE_HOST float saturate_fp8(float v)
{
    v = std::fma(v, 0.0f, v);
    if(std::isnan(v))
        return v;
    return v < -448.0f ? -448.0f : (v > 448.0f ? 448.0f : v);
}

/// f32 to FP8 E4M3 (OCP) bits, round to nearest even; |v| must not round past 448 (saturate
/// first). NaN becomes 0x7F with v's sign.
CK_TILE_HOST uint8_t float_to_fp8_e4m3_rne(float v)
{
    const uint8_t sign = std::signbit(v) ? 0x80 : 0;
    if(std::isnan(v))
        return sign | 0x7F;
    const double a = std::fabs(static_cast<double>(v));
    if(a < std::ldexp(1.0, -6))
        return sign | static_cast<uint8_t>(std::nearbyint(a / std::ldexp(1.0, -9)));
    int e    = std::ilogb(a);
    double q = std::nearbyint((a / std::ldexp(1.0, e) - 1.0) * 8.0);
    if(q == 8.0)
    {
        q = 0.0;
        ++e;
    }
    if(e > 8 || (e == 8 && q > 6.0))
        throw std::domain_error("float_to_fp8_e4m3_rne: " + std::to_string(v) + " past 448");
    return sign | static_cast<uint8_t>(((e + 7) << 3) | static_cast<int>(q));
}

/// f32 to f16 bits, round to nearest even (fptrunc): overflow to inf, NaN quieted.
CK_TILE_HOST uint16_t float_to_fp16_rne(float v)
{
    const uint16_t sign = std::signbit(v) ? 0x8000 : 0;
    if(std::isnan(v))
        return sign | 0x7E00 | static_cast<uint16_t>((float_bits(v) >> 13) & 0x1FF);
    const double a = std::fabs(static_cast<double>(v));
    if(std::isinf(v) || a >= 65520.0)
        return sign | 0x7C00;
    if(a < std::ldexp(1.0, -14))
        return sign | static_cast<uint16_t>(std::nearbyint(a / std::ldexp(1.0, -24)));
    int e    = std::ilogb(a);
    double q = std::nearbyint((a / std::ldexp(1.0, e) - 1.0) * 1024.0);
    if(q == 1024.0)
    {
        q = 0.0;
        ++e;
    }
    return sign | static_cast<uint16_t>(((e + 15) << 10) | static_cast<int>(q));
}

/// f32 to bf16 bits, round to nearest even (fptrunc): NaN quieted.
CK_TILE_HOST uint16_t float_to_bf16_rne(float v)
{
    const uint32_t u = float_bits(v);
    if(std::isnan(v))
        return static_cast<uint16_t>((u >> 16) | 0x40);
    return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

CK_TILE_HOST float fp16_bits_to_float(uint16_t h)
{
    const int exp  = (h >> 10) & 0x1F;
    const int mant = h & 0x3FF;
    float v        = 0.0f;
    if(exp == 0x1F)
        v = mant ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else
        v = exp == 0 ? std::ldexp(static_cast<float>(mant), -24)
                     : std::ldexp(1.0f + mant / 1024.0f, exp - 15);
    return (h & 0x8000) ? -v : v;
}

CK_TILE_HOST float bf16_bits_to_float(uint16_t h)
{
    return bits_float(static_cast<uint32_t>(h) << 16);
}

/// Bytes of one C or D element of O (fp8_t, half_t or bf16_t).
template <typename O>
CK_TILE_HOST constexpr index_t out_bytes()
{
    return std::is_same_v<O, fp8_t> ? 1 : 2;
}

/// C or D element i of `data`, of O, as f32.
template <typename O>
CK_TILE_HOST float load_out(const void* data, std::size_t i)
{
    if constexpr(std::is_same_v<O, fp8_t>)
        return fp8_e4m3_to_float(static_cast<const uint8_t*>(data)[i]);
    const uint16_t h = static_cast<const uint16_t*>(data)[i];
    return std::is_same_v<O, half_t> ? fp16_bits_to_float(h) : bf16_bits_to_float(h);
}

/// The epilogue's conversion of a final f32 value (FP8 saturated first) into element i of `data`,
/// of O.
template <typename O>
CK_TILE_HOST void store_out(void* data, std::size_t i, float v)
{
    if constexpr(std::is_same_v<O, fp8_t>)
        static_cast<uint8_t*>(data)[i] = float_to_fp8_e4m3_rne(saturate_fp8(v));
    else if constexpr(std::is_same_v<O, half_t>)
        static_cast<uint16_t*>(data)[i] = float_to_fp16_rne(v);
    else
    {
        static_assert(std::is_same_v<O, bf16_t>, "store_out: E is fp8, fp16 or bf16");
        static_cast<uint16_t*>(data)[i] = float_to_bf16_rne(v);
    }
}

} // namespace mx_tdm_scheduled

/**
 * @brief E = A @ B^T with K-major scales, as MxGemmPipelineAgBgCrCompTDMScheduled.
 *
 * The products are summed in double and rounded once to f32; with inputs whose f32 sums are
 * exact in any order (the tests' sparse check inputs) this is the kernel's accumulator bit for bit.
 * Then the output rounding (FP8 saturating).
 *
 * ADataType and BDataType are pk_fp4_t, fp8_t or bf8_t, ODataType fp8_t, half_t or bf16_t.
 * A and B are M (N) rows of row_bytes<T>(K) bytes, lda (ldb) bytes apart; the scales are
 * K-major (mx_scale_to_kmajor); E is row-major M x N in the output type, lde elements apart.
 * With `rows` set only those num_rows rows of E are computed, stored compactly: e's row i is
 * row rows[i] of E.
 */
template <typename ADataType, typename BDataType, typename ODataType>
CK_TILE_HOST void reference_mx_gemm_tdm_scheduled(const uint8_t* a,
                                                  long_index_t lda,
                                                  const uint8_t* b,
                                                  long_index_t ldb,
                                                  const uint8_t* a_scale,
                                                  const uint8_t* b_scale,
                                                  void* e,
                                                  long_index_t lde,
                                                  index_t M,
                                                  index_t N,
                                                  long_index_t K,
                                                  unsigned threads    = 0,
                                                  const index_t* rows = nullptr,
                                                  index_t num_rows    = 0)
{
    const index_t out_rows = rows ? num_rows : M;
    auto scale             = [](const uint8_t* s, index_t extent, index_t r, long_index_t k) {
        return mx_tdm_scheduled::e8m0_to_float(
            s[((k / mx_tdm_scheduled::kScaleStepK) * extent + r) * 4 +
              (k % mx_tdm_scheduled::kScaleStepK) / mx_tdm_scheduled::kScaleGroupK]);
    };
    auto row = [&](index_t i) {
        const index_t m = rows ? rows[i] : i;
        for(index_t n = 0; n < N; ++n)
        {
            double sum = 0.0;
            for(long_index_t k = 0; k < K; ++k)
            {
                const double x = static_cast<double>(
                                     mx_tdm_scheduled::decode_element<ADataType>(a + m * lda, k)) *
                                 scale(a_scale, M, m, k);
                const double y = static_cast<double>(
                                     mx_tdm_scheduled::decode_element<BDataType>(b + n * ldb, k)) *
                                 scale(b_scale, N, n, k);
                sum += x * y;
            }
            mx_tdm_scheduled::store_out<ODataType>(
                e, static_cast<std::size_t>(i) * lde + n, static_cast<float>(sum));
        }
    };
    if(threads == 0)
        threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<unsigned>(threads, static_cast<unsigned>(std::max<index_t>(out_rows, 1)));
    std::vector<std::thread> pool;
    for(unsigned t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for(index_t i = static_cast<index_t>(t); i < out_rows;
                i += static_cast<index_t>(threads))
                row(i);
        });
    for(auto& th : pool)
        th.join();
}

} // namespace ck_tile
