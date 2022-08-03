// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "ck/host_utility/io.hpp"

namespace ck {
namespace utils {

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value && !std::is_same<T, half_t>::value,
                        bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-5,
          double atol            = 3e-6)
{
    if(out.size() != ref.size())
    {
        std::cout << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        err = std::abs(out[i] - ref[i]);
        if(err > atol + rtol * std::abs(ref[i]) || !std::isfinite(out[i]) || !std::isfinite(ref[i]))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cout << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << out[i] << " != " << ref[i] << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cout << std::setw(12) << std::setprecision(7) << "max err: " << max_err << std::endl;
    }
    return res;
}

template <typename T>
typename std::enable_if<std::is_same<T, bhalf_t>::value, bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cout << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
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
        double o = type_convert<float>(out[i]);
        double r = type_convert<float>(ref[i]);
        err      = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cout << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cout << std::setw(12) << std::setprecision(7) << "max err: " << max_err << std::endl;
    }
    return res;
}

template <typename T>
typename std::enable_if<std::is_same<T, half_t>::value, bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cout << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<T>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        double o = type_convert<float>(out[i]);
        double r = type_convert<float>(ref[i]);
        err      = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cout << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cout << std::setw(12) << std::setprecision(7) << "max err: " << max_err << std::endl;
    }
    return res;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bhalf_t>::value, bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg = "Error: Incorrect results!",
          double                 = 0,
          double atol            = 0)
{
    if(out.size() != ref.size())
    {
        std::cout << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count   = 0;
    int64_t err     = 0;
    int64_t max_err = std::numeric_limits<int64_t>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        int64_t o = out[i];
        int64_t r = ref[i];
        err       = std::abs(o - r);

        if(err > atol)
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cout << msg << " out[" << i << "] != ref[" << i
                          << "]: " << static_cast<int>(out[i]) << " != " << static_cast<int>(ref[i])
                          << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cout << "max err: " << max_err << std::endl;
    }
    return res;
}

} // namespace utils
} // namespace ck
