// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief Host-side helpers for MX GEMM tests: 6-bit operand fillers and decoders, randomized E8M0
 *        scales, and a tolerance derived from the operand and output types and from K.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

namespace mx_test {

// ------------------------------------------------------------------------------------------
// 6-bit operands
//
// The GEMM stack stores FP6 (E2M3) and BF6 (E3M2) in the 12-byte legacy packs pk_fp6x16_t and
// pk_bf6x16_t. These have no converters for the generic fillers: the integer filler would write
// the same int into all three words, and FillUniformDistribution does not compile for them. As MX
// FlatMM does for FP6 (example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc), data are generated in
// the 16-byte pk_fp6_t / pk_bf6_t, which hold the same 16 x 6-bit LSB-first layout in their first
// 12 bytes, and the first sizeof(T) bytes of every element are copied into the legacy tensor.
// ------------------------------------------------------------------------------------------

template <typename T>
using mx_init_proxy_t = std::conditional_t<
    std::is_same_v<T, ck_tile::pk_fp6x16_t>,
    ck_tile::pk_fp6_t,
    std::conditional_t<std::is_same_v<T, ck_tile::pk_bf6x16_t>, ck_tile::pk_bf6_t, T>>;

/** @brief True for the 16-element legacy 6-bit packs pk_fp6x16_t and pk_bf6x16_t. */
template <typename T>
inline constexpr bool is_pk_f6x16_v =
    std::is_same_v<T, ck_tile::pk_fp6x16_t> || std::is_same_v<T, ck_tile::pk_bf6x16_t>;

/**
 * @brief Copies proxy-initialized data into the real tensor, keeping the first sizeof(DstT)
 *        bytes of every element (FlatMM's copy_proxy_to_real).
 */
template <typename DstT, typename SrcT>
void copy_proxy_to_real(ck_tile::HostTensor<DstT>& dst, const ck_tile::HostTensor<SrcT>& src)
{
    if constexpr(std::is_same_v<DstT, SrcT>)
    {
        dst = src;
    }
    else
    {
        static_assert(sizeof(DstT) <= sizeof(SrcT), "the proxy must be at least as large");
        static_assert(std::is_trivially_copyable_v<DstT> && std::is_trivially_copyable_v<SrcT>);
        static_assert(ck_tile::numeric_traits<DstT>::PackedSize ==
                          ck_tile::numeric_traits<SrcT>::PackedSize,
                      "the proxy must pack as many elements as the real type");
        if(dst.mData.size() != src.mData.size())
        {
            throw std::invalid_argument("copy_proxy_to_real: tensor sizes differ");
        }
        for(std::size_t i = 0; i < dst.mData.size(); ++i)
        {
            std::memcpy(&dst.mData[i], &src.mData[i], sizeof(DstT));
        }
    }
}

/**
 * @brief Fills an MX operand with values drawn uniformly from [lo, hi] and rounded to the
 *        operand type. The 6-bit legacy packs go through the pk_fp6_t / pk_bf6_t proxy.
 */
template <typename T>
void fill_mx_operand(ck_tile::HostTensor<T>& tensor, float lo, float hi, uint32_t seed)
{
    using Proxy = mx_init_proxy_t<T>;
    if constexpr(std::is_same_v<Proxy, T>)
    {
        ck_tile::FillUniformDistribution<T>{lo, hi, seed}(tensor);
    }
    else
    {
        ck_tile::HostTensor<Proxy> proxy(tensor.mDesc);
        ck_tile::FillUniformDistribution<Proxy>{lo, hi, seed}(proxy);
        copy_proxy_to_real(tensor, proxy);
    }
}

/**
 * @brief Decodes element i of a legacy 6-bit pack. The legacy decoder
 *        (pk_f6_legacy_t::fp6_e2m3_to_float) reads every pack as E2M3, so BF6 is decoded through
 *        the E3M2 rules of pk_bf6_t instead.
 */
template <typename T>
float pk_f6x16_to_float(const T& pack, ck_tile::index_t i)
{
    static_assert(is_pk_f6x16_v<T>, "expects pk_fp6x16_t or pk_bf6x16_t");
    const auto code = static_cast<uint8_t>(pack.unpack(i));
    return ck_tile::convert_to_float<mx_init_proxy_t<T>>(code);
}

/**
 * @brief Value of element (i0, i1) of a 2-D operand tensor, decoded to float without a scale.
 * @details For the 6-bit packs and pk_fp4_t, the pack and the position inside it follow the
 *          descriptor offset, as in reference_mx_gemm.
 */
template <typename T>
float mx_operand_value(const ck_tile::HostTensor<T>& tensor, std::size_t i0, std::size_t i1)
{
    if constexpr(is_pk_f6x16_v<T> || std::is_same_v<T, ck_tile::pk_fp4_t>)
    {
        constexpr std::size_t kPack = ck_tile::numeric_traits<T>::PackedSize;
        const std::size_t offset    = i0 * tensor.get_strides()[0] + i1 * tensor.get_strides()[1];
        if(offset / kPack >= tensor.mData.size())
        {
            throw std::out_of_range("mx_operand_value: element outside the packed storage");
        }
        const T& pack = tensor.mData[offset / kPack];
        if constexpr(std::is_same_v<T, ck_tile::pk_fp4_t>)
        {
            return offset % 2 == 0
                       ? ck_tile::type_convert<float>(pack.template unpack<>(ck_tile::number<0>{}))
                       : ck_tile::type_convert<float>(pack.template unpack<>(ck_tile::number<1>{}));
        }
        else
        {
            return pk_f6x16_to_float(pack, static_cast<ck_tile::index_t>(offset % kPack));
        }
    }
    else
    {
        static_assert(ck_tile::numeric_traits<T>::PackedSize == 1,
                      "packed types other than pk_fp4_t and the 6-bit packs are not handled here");
        return ck_tile::type_convert<float>(tensor(i0, i1));
    }
}

// ------------------------------------------------------------------------------------------
// E8M0 scales
// ------------------------------------------------------------------------------------------

/**
 * @brief Fills E8M0 scales with powers of two whose exponents are drawn uniformly and
 *        independently per element from [exp_lo, exp_hi].
 * @details Neighbouring scale blocks therefore differ, which constant scales would hide. Codes
 *          0x00 (whose host decode is 2^-126, not 2^-127) and 0xFF (NaN) are never produced.
 */
inline void fill_random_e8m0(ck_tile::HostTensor<ck_tile::e8m0_t>& scales,
                             int exp_lo,
                             int exp_hi,
                             uint32_t seed)
{
    if(!(-126 <= exp_lo && exp_lo <= exp_hi && exp_hi <= 127))
    {
        throw std::invalid_argument("fill_random_e8m0: need -126 <= exp_lo <= exp_hi <= 127");
    }
    ck_tile::FillUniformScaleDistribution<ck_tile::e8m0_t>{
        std::ldexp(1.f, exp_lo), std::ldexp(1.f, exp_hi), seed}(scales);
}

/**
 * @brief Exponent range for fill_random_e8m0 that cannot overflow the output.
 * @details With |a| <= a_max, |b| <= b_max and both scales at most 2^hi, every output is bounded
 *          by K * a_max * b_max * 2^(2 * hi). hi is the largest exponent that keeps this bound at
 *          or below half of c_max, and lo = hi - span, so the range holds span + 1 distinct
 *          scales. hi is clamped to [-126 + span, 127]; if it is clamped up, the bound no longer
 *          holds.
 */
inline std::pair<int, int>
e8m0_exponent_range(ck_tile::index_t K, float a_max, float b_max, float c_max, int span = 4)
{
    if(K <= 0 || !(a_max > 0) || !(b_max > 0) || !(c_max > 0) || span < 0 || span > 253)
    {
        throw std::invalid_argument("e8m0_exponent_range: need K, a_max, b_max, c_max > 0 and "
                                    "0 <= span <= 253");
    }
    const double bound   = static_cast<double>(K) * a_max * b_max;
    const int hi         = static_cast<int>(std::floor(0.5 * std::log2(0.5 * c_max / bound)));
    const int hi_clamped = std::clamp(hi, -126 + span, 127);
    return {hi_clamped - span, hi_clamped};
}

// ------------------------------------------------------------------------------------------
// Tolerance
// ------------------------------------------------------------------------------------------

/**
 * @brief Mantissa bits of an MX operand or output type. The legacy 6-bit packs have no
 *        numeric_traits::mant, so they use their proxy: E2M3 has 3, E3M2 has 2.
 */
template <typename T>
inline constexpr int mx_mantissa_bits = ck_tile::numeric_traits<mx_init_proxy_t<T>>::mant;

struct MxTolerance
{
    double rtol;
    double atol;
};

/**
 * @brief Tolerance for comparing an MX GEMM output against reference_mx_gemm, for check_err's
 *        test |out - ref| <= atol + rtol * |ref|.
 * @details
 *   rtol = 2^-mant(C)
 *       Device and host each round the fp32 result to C once. Assuming both round to nearest
 *       (true in gfx950 and gfx12 builds, where the host uses the native bf16 conversion), two
 *       roundings of nearly equal values differ by at most one ULP of C, and one ULP is at most
 *       2^-mant(C) * |ref|.
 *   atol = 2 * K * 2^-24 * S_max
 *       Both sides accumulate K products in fp32, in different orders. Each sum is within about
 *       K * 2^-24 * S of the exact value (the first-order gamma_K bound), so the two differ by at
 *       most about twice that. This assumes the device's scaled WMMA accumulates in fp32.
 *     + (2^-(mant(A) + 1) + 2^-(mant(B) + 1)) * S_max      only if quantized_reference is false
 *       A reference computed from the unquantized values also sees the input rounding of A and B,
 *       half a ULP each for normal values (subnormal inputs are not covered). reference_mx_gemm
 *       decodes the same codes and scales as the kernel, so its input rounding is identical and
 *       cancels; this does not hold for BF6 yet, which reference_mx_gemm decodes as E2M3.
 *   S_max = max over (m, n) of sum_k |a(m, k) * b(k, n) * scale_a * scale_b|.
 */
template <typename ADataType, typename BDataType, typename CDataType>
MxTolerance mx_gemm_tolerance(ck_tile::index_t K, double s_max, bool quantized_reference = true)
{
    const double rtol = std::ldexp(1.0, -mx_mantissa_bits<CDataType>);
    double atol       = 2.0 * K * std::ldexp(1.0, -24) * s_max;
    if(!quantized_reference)
    {
        atol += (std::ldexp(1.0, -(mx_mantissa_bits<ADataType> + 1)) +
                 std::ldexp(1.0, -(mx_mantissa_bits<BDataType> + 1))) *
                s_max;
    }
    return {rtol, atol};
}

/**
 * @brief S_max for mx_gemm_tolerance, for the reference_mx_gemm layouts: a (M, K), b (K, N),
 *        scale_a (M, K / ScaleBlockSize), scale_b (K / ScaleBlockSize, N).
 * @details Each operand is decoded and scaled once, into contiguous K-major rows, rather than once
 *          per output element, and the M rows of sums run in parallel. Operand values and scales
 *          have few mantissa bits, so each product |a * scale_a| * |b * scale_b| is exact in double
 *          however its four factors are grouped.
 * @throws std::invalid_argument if a tensor is not 2-D, scale_block_size is not positive, b does
 *         not have K rows, or a scale tensor is smaller than its operand needs (scale_a at least
 *         M x ceil(K / scale_block_size), scale_b at least ceil(K / scale_block_size) x N). The
 *         checks are here because HostTensor::operator() does not check bounds.
 */
template <typename ADataType, typename BDataType, typename AScaleDataType, typename BScaleDataType>
double mx_sum_abs_max(const ck_tile::HostTensor<ADataType>& a,
                      const ck_tile::HostTensor<BDataType>& b,
                      const ck_tile::HostTensor<AScaleDataType>& scale_a,
                      const ck_tile::HostTensor<BScaleDataType>& scale_b,
                      ck_tile::index_t scale_block_size)
{
    if(a.get_num_of_dimension() != 2 || b.get_num_of_dimension() != 2 ||
       scale_a.get_num_of_dimension() != 2 || scale_b.get_num_of_dimension() != 2 ||
       scale_block_size <= 0)
    {
        throw std::invalid_argument("mx_sum_abs_max: needs 2-D tensors and scale_block_size > 0");
    }
    const std::size_t M     = a.get_lengths()[0];
    const std::size_t K     = a.get_lengths()[1];
    const std::size_t N     = b.get_lengths()[1];
    const std::size_t block = static_cast<std::size_t>(scale_block_size);
    const std::size_t KB    = (K + block - 1) / block;
    if(b.get_lengths()[0] != K || scale_a.get_lengths()[0] < M || scale_a.get_lengths()[1] < KB ||
       scale_b.get_lengths()[0] < KB || scale_b.get_lengths()[1] < N)
    {
        throw std::invalid_argument("mx_sum_abs_max: operand and scale shapes do not match");
    }

    std::vector<double> a_abs(M * K);
    std::vector<double> b_abs(N * K);
    for(std::size_t m = 0; m < M; ++m)
    {
        for(std::size_t k = 0; k < K; ++k)
        {
            a_abs[m * K + k] = std::abs(static_cast<double>(mx_operand_value(a, m, k)) *
                                        ck_tile::type_convert<float>(scale_a(m, k / block)));
        }
    }
    for(std::size_t n = 0; n < N; ++n)
    {
        for(std::size_t k = 0; k < K; ++k)
        {
            b_abs[n * K + k] = std::abs(static_cast<double>(mx_operand_value(b, k, n)) *
                                        ck_tile::type_convert<float>(scale_b(k / block, n)));
        }
    }

    // Each m is summed by one thread, which writes only row_max[m]. Nothing below may throw: an
    // exception that escapes a worker thread calls std::terminate, so all decoding stays above.
    std::vector<double> row_max(M, 0.0);
    auto sum_row = [&](std::size_t m) {
        const double* a_row = a_abs.data() + m * K;
        double best         = 0.0;
        for(std::size_t n = 0; n < N; ++n)
        {
            const double* b_row = b_abs.data() + n * K;
            double s            = 0.0;
            for(std::size_t k = 0; k < K; ++k)
            {
                s += a_row[k] * b_row[k];
            }
            best = std::max(best, s);
        }
        row_max[m] = best;
    };
    // ParallelTensorFunctor divides by the thread count, which must therefore be at least 1. More
    // threads than rows would only idle, and the cap of 64 keeps parallel test binaries on a
    // many-core host from oversubscribing it.
    const std::size_t num_threads = std::max<std::size_t>(
        1, std::min<std::size_t>({M, ck_tile::get_available_cpu_cores(), std::size_t{64}}));
    ck_tile::make_ParallelTensorFunctor(sum_row, M)(num_threads);
    return M == 0 ? 0.0 : *std::max_element(row_max.begin(), row_max.end());
}

} // namespace mx_test
