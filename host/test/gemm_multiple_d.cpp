#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include <algorithm>
#include <iterator>
#include <test.hpp>
#include <rtc/compile_kernel.hpp>

std::vector<rtc::src_file> get_headers_for_test()
{
    std::vector<rtc::src_file> result;
    auto hs = ck::host::GetHeaders();
    std::transform(
        hs.begin(), hs.end(), std::back_inserter(result), [&](const auto& p) -> rtc::src_file {
            auto s = p.second;
            std::string content{s.first, s.second};
            return {p.first, content};
        });
    return result;
}

const std::string gemm_compile_check = R"__ck__(
#include <${include}>

extern "C" __global__ void f(const ck::half_t* a, const ck::half_t* b, ck::half_t* c) {
    using G = ${template};
    constexpr auto desc = ${template}::make_descriptor(ck::make_naive_tensor_descriptor_packed(ck::make_tuple(${m}, ${k})),
                                             ck::make_naive_tensor_descriptor_packed(ck::make_tuple(${n}, ${k})),
                                             ck::make_tuple(),
                                             ck::make_naive_tensor_descriptor_packed(ck::make_tuple(${m}, ${n})));

    static_assert(desc.IsValid(), "Invalid ck gemm.");

    if constexpr(desc.IsValid())
    {
        ${template}::Run(desc,
               a,
               b,
               ck::make_tuple(),
               c);
    }
}

)__ck__";

TEST_CASE(test_problem_kernel)
{
    ck::host::device_gemm_multiple_d::Problem prob;
    prob.M = 1024;
    prob.N = 1024;
    prob.K = 1024;
    for(auto solution : prob.GetSolutions("gfx90a"))
    {
        auto src  = ck::host::InterpolateString(gemm_compile_check,
                                               {{"include", prob.GetIncludeHeader()},
                                                {"template", solution.ToTemplateString()},
                                                {"m", std::to_string(prob.M)},
                                                {"n", std::to_string(prob.N)},
                                                {"k", std::to_string(prob.K)}});
        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "f";
        rtc::compile_kernel(srcs, options);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
