#pragma once
#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <random>
#include <test.hpp>
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <fstream>
#include <unordered_set>
#include "ck/host/headers.hpp"
#include "rtc/hiprtc_enable_env.hpp"
#include "ck/host/stringutils.hpp"

// NOLINTNEXTLINE
const char* const disable_warning_pragma = R"__migraphx__(
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
${content}
#pragma clang diagnostic pop
)__migraphx__";

template <class P>
inline std::string ck_disable_warnings(P p)
{
    return ck::host::InterpolateString(disable_warning_pragma,
                                       {{"content", std::string{p.data(), p.size()}}});
}

inline std::vector<rtc::src_file> create_headers_for_hiprtc_test()
{
    auto ck_headers = ck::host::GetHeaders();
    std::vector<rtc::src_file> result;

    std::transform(ck_headers.begin(), ck_headers.end(), std::back_inserter(result), [&](auto& p) {
        return rtc::src_file{p.first, ck_disable_warnings(p.second)};
    });

    return result;
}

inline const std::vector<rtc::src_file>& get_headers_for_hiprtc_test()
{
    static const std::vector<rtc::src_file> headers = create_headers_for_hiprtc_test();
    return headers;
}

inline std::vector<rtc::src_file> create_headers_for_clang_test()
{
    std::vector<rtc::src_file> result;
    auto hs = ck::host::GetHeaders();
    std::transform(
        hs.begin(), hs.end(), std::back_inserter(result), [&](const auto& p) -> rtc::src_file {
            return {p.first, {p.second.begin(), p.second.end()}};
        });
    return result;
}

inline const std::vector<rtc::src_file>& get_headers_for_clang_test()
{
    static const std::vector<rtc::src_file> headers = create_headers_for_clang_test();
    return headers;
}

inline const std::vector<rtc::src_file>& get_headers_for_test()
{
    if(ck::EnvIsEnabled(CK_ENV(CK_CODEGEN_TESTS_ENABLE_HIPRTC)))
    {
        return get_headers_for_hiprtc_test();
    }
    else
    {
        return get_headers_for_clang_test();
    }
}

template <typename V>
std::size_t GetSize(V mLens, V mStrides)
{
    std::size_t space = 1;
    for(std::size_t i = 0; i < mLens.Size(); ++i)
    {
        if(mLens[i] == 0)
            continue;

        space += (mLens[i] - 1) * mStrides[i];
    }
    return space;
}

template <class T>
rtc::buffer<T> generate_buffer(std::size_t n, std::size_t seed = 0)
{
    rtc::buffer<T> result(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0);
    std::generate(result.begin(), result.end(), [&] { return dis(gen); });
    return result;
}

template <class T, typename V>
std::enable_if_t<!std::is_integral_v<V>, rtc::buffer<T>>
generate_buffer(V mLens, V mStrides, std::size_t seed = 0)
{
    std::size_t space = GetSize(mLens, mStrides);
    return generate_buffer<T>(space, seed);
}

template <class T, class U>
bool allclose(const T& a, const U& b, double atol = 0.01, double rtol = 0.01)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [&](double x, double y) {
        return fabs(x - y) < atol + rtol * fabs(y);
    });
}

std::string classify(double x)
{
    switch(std::fpclassify(x))
    {
    case FP_INFINITE: return "inf";
    case FP_NAN: return "nan";
    case FP_NORMAL: return "normal";
    case FP_SUBNORMAL: return "subnormal";
    case FP_ZERO: return "zero";
    default: return "unknown";
    }
}

template <class Buffer>
void print_classification(const Buffer& x)
{
    std::unordered_set<std::string> result;
    for(const auto& i : x)
        result.insert(classify(i));
    for(const auto& c : result)
        std::cout << c << ", ";
    std::cout << std::endl;
}

template <class Buffer>
void print_statistics(const Buffer& x)
{
    std::cout << "Min value: " << *std::min_element(x.begin(), x.end()) << ", ";
    std::cout << "Max value: " << *std::max_element(x.begin(), x.end()) << ", ";
    double num_elements = x.size();
    auto mean =
        std::accumulate(x.begin(), x.end(), double{0.0}, std::plus<double>{}) / num_elements;
    auto stddev = std::sqrt(
        std::accumulate(x.begin(),
                        x.end(),
                        double{0.0},
                        [&](double r, double v) { return r + std::pow((v - mean), 2.0); }) /
        num_elements);
    std::cout << "Mean: " << mean << ", ";
    std::cout << "StdDev: " << stddev << "\n";
}

template <class Buffer>
void print_preview(const Buffer& x)
{
    if(x.size() <= 10)
    {
        std::for_each(x.begin(), x.end(), [&](double i) { std::cout << i << ", "; });
    }
    else
    {
        std::for_each(x.begin(), x.begin() + 5, [&](double i) { std::cout << i << ", "; });
        std::cout << "..., ";
        std::for_each(x.end() - 5, x.end(), [&](double i) { std::cout << i << ", "; });
    }
    std::cout << std::endl;
}

template <class T>
struct check_all
{
    rtc::buffer<T> data{};
    bool operator()(const rtc::buffer<T>& x)
    {
        if(data.empty())
        {
            data = x;
            return true;
        }
        return allclose(data, x);
    }
};

template <class Solution>
auto report(const Solution& solution, bool pass)
{
    return test::make_predicate(solution.ToTemplateString(), [=] { return pass; });
}
