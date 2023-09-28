#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include <algorithm>
#include <iterator>
#include <test.hpp>
#include <rtc/compile_kernel.hpp>

const std::string compile_check = R"__ck__(
#include <${include}>

extern "C" __global__ void f() {
    using type = ${template}::DeviceOp;
}

)__ck__";

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

TEST_CASE(test_operation)
{
    ck::host::device_gemm_multiple_d::Problem prob;
    prob.M   = 256;
    prob.N   = 256;
    prob.K   = 256;
    auto ops = ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle::CreateOperations(prob);
    for(auto op : ops)
    {
        auto solution = op.ToSolution();
        std::string include =
            "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp";
        auto src = ck::host::InterpolateString(
            compile_check, {{"include", include}, {"template", solution.ToTemplateString()}});
        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "f";
        rtc::compile_kernel(srcs, options);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
