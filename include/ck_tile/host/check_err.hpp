// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file
 * @brief Error checking utilities for numerical computations and type conversions
 *
 * This file provides utilities for checking numerical errors between computed and reference results,
 * handling various data types including floating point (FP8, BF8, FP16, BF16, FP32) and integer types.
 * It includes functions for calculating error thresholds and comparing results with specified tolerances.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/ranges.hpp"

namespace ck_tile {

/** @brief 8-bit floating point type */
using F8   = ck_tile::fp8_t;
/** @brief 8-bit brain floating point type */
using BF8  = ck_tile::bf8_t;
/** @brief 16-bit floating point (half precision) type */
using F16  = ck_tile::half_t;
/** @brief 16-bit brain floating point type */
using BF16 = ck_tile::bf16_t;
/** @brief 32-bit floating point (single precision) type */
using F32  = float;
/** @brief 8-bit signed integer type */
using I8   = int8_t;
/** @brief 32-bit signed integer type */
using I32  = int32_t;

/**
 * @brief Calculate relative error threshold for numerical comparisons
 *
 * @tparam ComputeDataType Type used for computation
 * @tparam OutDataType Type used for output
 * @tparam AccDataType Type used for accumulation (defaults to ComputeDataType)
 * @param number_of_accumulations Number of accumulation operations performed
 * @return double Relative error threshold based on data type characteristics
 */
template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
double get_relative_threshold(const int number_of_accumulations = 1)
{

    static_assert(
        is_any_of<ComputeDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
        "Warning: Unhandled ComputeDataType for setting up the relative threshold!");

    double compute_error = 0;
    if constexpr(is_any_of<ComputeDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        compute_error = std::pow(2, -numeric_traits<ComputeDataType>::mant) * 0.5;
    }

    static_assert(is_any_of<OutDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
                  "Warning: Unhandled OutDataType for setting up the relative threshold!");

    double output_error = 0;
    if constexpr(is_any_of<OutDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        output_error = std::pow(2, -numeric_traits<OutDataType>::mant) * 0.5;
    }
    double midway_error = std::max(compute_error, output_error);

    static_assert(is_any_of<AccDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
                  "Warning: Unhandled AccDataType for setting up the relative threshold!");

    double acc_error = 0;
    if constexpr(is_any_of<AccDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        acc_error = std::pow(2, -numeric_traits<AccDataType>::mant) * 0.5 * number_of_accumulations;
    }
    return std::max(acc_error, midway_error);
}

/**
 * @brief Calculate absolute error threshold for numerical comparisons
 *
 * @tparam ComputeDataType Type used for computation
 * @tparam OutDataType Type used for output
 * @tparam AccDataType Type used for accumulation (defaults to ComputeDataType)
 * @param max_possible_num Maximum possible value in the computation
 * @param number_of_accumulations Number of accumulation operations performed
 * @return double Absolute error threshold based on data type characteristics and maximum value
 */
template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
double get_absolute_threshold(const double max_possible_num, const int number_of_accumulations = 1)
{
    static_assert(
        is_any_of<ComputeDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
        "Warning: Unhandled ComputeDataType for setting up the absolute threshold!");

    auto expo            = std::log2(std::abs(max_possible_num));
    double compute_error = 0;
    if constexpr(is_any_of<ComputeDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        compute_error = std::pow(2, expo - numeric_traits<ComputeDataType>::mant) * 0.5;
    }

    static_assert(is_any_of<OutDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
                  "Warning: Unhandled OutDataType for setting up the absolute threshold!");

    double output_error = 0;
    if constexpr(is_any_of<OutDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        output_error = std::pow(2, expo - numeric_traits<OutDataType>::mant) * 0.5;
    }
    double midway_error = std::max(compute_error, output_error);

    static_assert(is_any_of<AccDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
                  "Warning: Unhandled AccDataType for setting up the absolute threshold!");

    double acc_error = 0;
    if constexpr(is_any_of<AccDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        acc_error =
            std::pow(2, expo - numeric_traits<AccDataType>::mant) * 0.5 * number_of_accumulations;
    }
    return std::max(acc_error, midway_error);
}

/**
 * @brief Stream operator overload for vector output
 *
 * @tparam T Type of vector elements
 * @param os Output stream
 * @param v Vector to output
 * @return std::ostream& Reference to the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    using size_type = typename std::vector<T>::size_type;

    os << "[";
    for(size_type idx = 0; idx < v.size(); ++idx)
    {
        if(0 < idx)
        {
            os << ", ";
        }
        os << v[idx];
    }
    return os << "]";
}

/**
 * @brief Check for size mismatch between output and reference ranges
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if sizes mismatch
 * @return bool True if sizes mismatch, false otherwise
 */
template <typename Range, typename RefRange>
bool check_size_mismatch(const Range& out,
                         const RefRange& ref,
                         const std::string& msg = "Error: Incorrect results!")
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return true;
    }
    return false;
}

/**
 * @brief Report error statistics for numerical comparisons
 *
 * @param err_count Number of errors found
 * @param max_err Maximum error value encountered
 * @param total_size Total number of elements compared
 */
void report_error_stats(int err_count, double max_err, std::size_t total_size)
{
    const float error_percent =
        static_cast<float>(err_count) / static_cast<float>(total_size) * 100.f;
    std::cerr << "max err: " << max_err;
    std::cerr << ", number of errors: " << err_count;
    std::cerr << ", " << error_percent << "% wrong values" << std::endl;
}

/**
 * @brief Check errors between floating point ranges with specified tolerances
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return bool True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_floating_point_v<ranges::range_value_t<Range>> &&
        !std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type CK_TILE_HOST
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg  = "Error: Incorrect results!",
          double rtol             = 1e-5,
          double atol             = 3e-6,
          bool allow_infinity_ref = false)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = *std::next(std::begin(out), i);
        const double r = *std::next(std::begin(ref), i);
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

/**
 * @brief Check errors between half precision floating point ranges
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return bool True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type CK_TILE_HOST
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg  = "Error: Incorrect results!",
          double rtol             = 1e-3,
          double atol             = 1e-3,
          bool allow_infinity_ref = false)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = static_cast<double>(std::numeric_limits<ranges::range_value_t<Range>>::min());
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

/**
 * @brief Check errors between integer ranges
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param rtol Relative tolerance (unused for integers)
 * @param atol Absolute tolerance
 * @return bool True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_integral_v<ranges::range_value_t<Range>> &&
                  !std::is_same_v<ranges::range_value_t<Range>, bf16_t>)
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                     || std::is_same_v<ranges::range_value_t<Range>, int4_t>
#endif
                 ,
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                          const RefRange& ref,
                          const std::string& msg = "Error: Incorrect results!",
                          double                 = 0,
                          double atol            = 0)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    bool res{true};
    int err_count   = 0;
    int64_t err     = 0;
    int64_t max_err = std::numeric_limits<int64_t>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const int64_t o = *std::next(std::begin(out), i);
        const int64_t r = *std::next(std::begin(ref), i);
        err             = std::abs(o - r);

        if(err > atol)
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << " out[" << i << "] != ref[" << i << "]: " << o << " != " << r
                          << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, static_cast<double>(max_err), ref.size());
    }
    return res;
}

/**
 * @brief Check errors between FP8 ranges
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param max_rounding_point_distance Maximum allowed distance between rounding points
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return bool True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, fp8_t>),
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                          const RefRange& ref,
                          const std::string& msg               = "Error: Incorrect results!",
                          unsigned max_rounding_point_distance = 1,
                          double atol                         = 1e-1,
                          bool allow_infinity_ref             = false)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    static const auto get_rounding_point_distance = [](fp8_t o, fp8_t r) -> unsigned {
        static const auto get_sign_bit = [](fp8_t v) -> bool {
            return 0x80 & bit_cast<uint8_t>(v);
        };

        if(get_sign_bit(o) ^ get_sign_bit(r))
        {
            return std::numeric_limits<unsigned>::max();
        }
        else
        {
            return std::abs(bit_cast<int8_t>(o) - bit_cast<int8_t>(r));
        }
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const fp8_t o_fp8   = *std::next(std::begin(out), i);
        const fp8_t r_fp8   = *std::next(std::begin(ref), i);
        const double o_fp64 = type_convert<float>(o_fp8);
        const double r_fp64 = type_convert<float>(r_fp8);
        err                 = std::abs(o_fp64 - r_fp64);
        if(!(less_equal<double>{}(err, atol) ||
             get_rounding_point_distance(o_fp8, r_fp8) <= max_rounding_point_distance) ||
           is_infinity_error(o_fp64, r_fp64))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o_fp64 << " != " << r_fp64 << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

/**
 * @brief Check errors between BF8 ranges
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return bool True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, bf8_t>),
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                          const RefRange& ref,
                          const std::string& msg  = "Error: Incorrect results!",
                          double rtol             = 1e-3,
                          double atol             = 1e-3,
                          bool allow_infinity_ref = false)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

} // namespace ck_tile
