// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type.hpp"
#include "ck/host_utility/io.hpp"

#include "ck/library/utility/ranges.hpp"

namespace ck {
namespace utils {

template <typename ComputeDataType>
double get_relative_threshold()
{
    using F8   = ck::f8_t;
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32  = float;

    int mantissa = 0;
    if constexpr(is_same_v<ComputeDataType, F8>)
    {
        mantissa = NumericUtils<F8>::mant;
    }
    else if constexpr(is_same_v<ComputeDataType, F16>)
    {
        mantissa = NumericUtils<F16>::mant;
    }
    else if constexpr(is_same_v<ComputeDataType, BF16>)
    {
        mantissa = NumericUtils<BF16>::mant;
    }
    else if constexpr(is_same_v<ComputeDataType, F32>)
    {
        mantissa = NumericUtils<F32>::mant;
    }
    else
    {
        std::cout << "Warning: Unhandled ComputeDataType for setting up the relative threshold!"
                  << std::endl;
    }

    return std::pow(2, -mantissa) * 0.5;
}

template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_floating_point_v<ranges::range_value_t<Range>> &&
        !std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-5,
          double atol            = 3e-6)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = *std::next(std::begin(out), i);
        const double r = *std::next(std::begin(ref), i);
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
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
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, bhalf_t>,
    bool>::type
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count = 0;
    double err    = 0;
    // TODO: This is a hack. We should have proper specialization for bhalf_t data type.
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
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
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = NumericLimits<ranges::range_value_t<Range>>::Min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
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
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_integral_v<ranges::range_value_t<Range>> &&
                  !std::is_same_v<ranges::range_value_t<Range>, bhalf_t> &&
                  !std::is_same_v<ranges::range_value_t<Range>, f8_t> &&
                  !std::is_same_v<ranges::range_value_t<Range>, bf8_t>)
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                     || std::is_same_v<ranges::range_value_t<Range>, int4_t>
#endif
                 ,
                 bool>
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double                 = 0,
          double atol            = 0)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

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
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, f8_t>),
                 bool>
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    double err_acc = 0;

    std::cout << "ref size: " << ref.size() << '\n';
    std::cout << "f8_t exp value: " << NumericUtils<f8_t>::exp << '\n';
    std::cout << "f8_t mantissa bits: " << NumericUtils<f8_t>::mant << std::endl;

    // cout sizeof - digits (that is mantissa bits) = exponent bits
    // std::cout << "int8_t exp value: " << std::numeric_limits<int8_t>::max_exponent << '\n';
    // std::cout << "int8_t mantissa bits: " << std::numeric_limits<int8_t>::digits << std::endl;

    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        err_acc += err;

        // atol = err_acc * 2. * (NumericUtils<f8_t>::exp - NumericUtils<f8_t>::mant);
        // rtol = err_acc * std::pow(2., -NumericUtils<f8_t>::mant);

        if(i % (ref.size() / 10) == 0)
        {
            std::cout << "current error at [" << i << "]: " << err << '\n';
            std::cout << "err acc [" << i << "]: " << err_acc << '\n';
            std::cout << "abs(r): " << std::abs(r) << '\n';
            std::cout << "atol: " << atol << '\n';
            std::cout << "rtol: " << rtol << std::endl;
        }

        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
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
    std::cout << "f8_t err_acc: " << err_acc << std::endl;
    if(!res)
    {
        std::cerr << std::setw(12) << std::setprecision(7) << "max err: " << max_err
                  << " number of errors: " << err_count << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, bf8_t>),
                 bool>
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
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
        std::cerr << std::setw(12) << std::setprecision(7) << "max err: " << max_err << std::endl;
    }
    return res;
}

} // namespace utils
} // namespace ck
