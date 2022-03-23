#ifndef CHECK_ERR_HPP
#define CHECK_ERR_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <half.hpp>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include "data_type.hpp"

namespace ck {
namespace utils {

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value && !std::is_same<T, half_t>::value,
                        bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-5,
          double atol            = 1e-8)
{
    if(out.size() != ref.size())
    {
        std::cout << "out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl
                  << msg << std::endl;
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
                std::cout << std::setw(12) << std::setprecision(7) << "out[" << i << "] != ref["
                          << i << "]: " << out[i] << " != " << ref[i] << std::endl
                          << msg << std::endl;
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
        std::cout << "out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl
                  << msg << std::endl;
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
                std::cout << std::setw(12) << std::setprecision(7) << "out[" << i << "] != ref["
                          << i << "]: " << o << " != " << r << std::endl
                          << msg << std::endl;
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
typename std::enable_if<std::is_same<T, half_t>::value || std::is_same<T, half_float::half>::value,
                        bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cout << "out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl
                  << msg << std::endl;
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
                std::cout << std::setw(12) << std::setprecision(7) << "out[" << i << "] != ref["
                          << i << "]: " << o << " != " << r << std::endl
                          << msg << std::endl;
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
          double                 = 0)
{
    if(out.size() != ref.size())
    {
        std::cout << "out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl
                  << msg << std::endl;
        return false;
    }

    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        if(out[i] != ref[i])
        {
            std::cout << "out[" << i << "] != ref[" << i << "]: " << out[i] << " != " << ref[i]
                      << std::endl
                      << msg << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace utils
} // namespace ck

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(os, " "));
    return os;
}

#endif
