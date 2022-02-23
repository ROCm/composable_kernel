#ifndef TEST_UTIL_HPP
#define TEST_UTIL_HPP

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <type_traits>
#include <vector>

namespace test_util {

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
check_err(const std::vector<T>& out,
          const std::vector<T>& ref,
          const std::string& msg,
          T rtol = static_cast<T>(1e-5),
          T atol = static_cast<T>(1e-8))
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
    T err         = 0;
    T max_err     = std::numeric_limits<T>::min();
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
                          << i << "]: " << out[i] << "!=" << ref[i] << std::endl
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
typename std::enable_if<std::is_integral<T>::value, bool>::type check_err(
    const std::vector<T>& out, const std::vector<T>& ref, const std::string& msg, T = 0, T = 0)
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
            std::cout << "out[" << i << "] != ref[" << i << "]: " << out[i] << "!=" << ref[i]
                      << std::endl
                      << msg << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace test_util

#endif
