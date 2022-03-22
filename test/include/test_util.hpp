#ifndef TEST_UTIL_HPP
#define TEST_UTIL_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include "data_type.hpp"

namespace test {

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value && !std::is_same<T, ck::half_t>::value,
                        bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg,
          double rtol = 1e-5,
          double atol = 1e-8)
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
typename std::enable_if<std::is_same<T, ck::bhalf_t>::value || std::is_same<T, ck::half_t>::value,
                        bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg,
          double rtol = 1e-5,
          double atol = 1e-8)
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
    double max_err = ck::type_convert<float>(ck::NumericLimits<T>::Min());
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        float o = ck::type_convert<float>(out[i]);
        float r = ck::type_convert<float>(ref[i]);
        err     = std::abs(o - r);
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

bool check_err(const std::vector<_Float16>& out,
                   const std::vector<_Float16>& ref,
                   const std::string& msg,
                   _Float16 rtol = static_cast<_Float16>(1e-3f),
                   _Float16 atol = static_cast<_Float16>(1e-3f))
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
    double err         = 0;
    double max_err     = std::numeric_limits<_Float16>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        double out_ = double(out[i]);
        double ref_ = double(ref[i]);
        err = std::abs(out_ - ref_);
        if(err > atol + rtol * std::abs(ref_) || !std::isfinite(out_) || !std::isfinite(ref_))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cout << std::setw(12) << std::setprecision(7) << "out[" << i << "] != ref["
                          << i << "]: " << out_ << "!=" << ref_ << std::endl
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
typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, ck::bhalf_t>::value,
                        bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg,
          double = 0,
          double = 0)
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

} // namespace test

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(os, " "));
    return os;
}

#endif
