#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/host/prototype_header.hpp"
#include "ck/host/mgx_instances.hpp"
#include "common.hpp"
#include <iostream>
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <test.hpp>
#include <cmath>
#include <typeindex>

using half = _Float16;

using namespace ck::host;

struct GemmProblem
{
    int M                = 0;
    int N                = 0;
    int K                = 0;
    Layout ALayout       = Layout::RowMajor;
    Layout BLayout       = Layout::RowMajor;
    Layout CLayout       = Layout::RowMajor;
    DataType ADataType   = DataType::Half;
    DataType BDataType   = DataType::Half;
    DataType CDataType   = DataType::Half;
    DataType AccDataType = DataType::Float;
    bool TransA          = false;
    bool TransB          = false;
    bool TransC          = false;
};

struct GemmSolution
{
    std::string kernel;
    GemmKernelInstanceParams instance_params;
};

auto GetInstanceArray(DataType a_type, DataType b_type, DataType c_type)
{
    if(a_type == DataType::Half && b_type == DataType::Half && c_type == DataType::Half)
    {
        return std::make_tuple(fp16_fp16_fp16_instance_params.data(),
                               fp16_fp16_fp16_instance_params.size());
    }
    else if(a_type == DataType::BF16 && b_type == DataType::BF16 && c_type == DataType::BF16)
    {
        return std::make_tuple(bf16_bf16_bf16_instance_params.data(),
                               bf16_bf16_bf16_instance_params.size());
    }
    else if(a_type == DataType::FP8 && b_type == DataType::FP8 && c_type == DataType::Half)
    {
        return std::make_tuple(fp8_fp8_fp16_instance_params.data(),
                               fp8_fp8_fp16_instance_params.size());
    }
    else if(a_type == DataType::BF8 && b_type == DataType::BF8 && c_type == DataType::Half)
    {
        return std::make_tuple(bf8_bf8_fp16_instance_params.data(),
                               bf8_bf8_fp16_instance_params.size());
    }
    else
    {
        throw std::runtime_error("Invalid gemm input type configuration");
    }
}

std::vector<GemmSolution> GetGemmSolutions(const GemmProblem& problem)
{
    static const std::string template_string = R"__ck__(
    ck_tile::MGXPrototypeGemmKernel<${BaseGemmPipeline},
                                    ${GemmPipeline},
                                    ${DoubleSmemBuffer},
                                    ${Scheduler},
                                    ${EpilogueSelector},
                                    ${TileM},
                                    ${TileN},
                                    ${TileK},
                                    ${WarpM},
                                    ${WarpN},
                                    ${WarpK},
                                    ${WarpTileM},
                                    ${WarpTileN},
                                    ${WarpTileK},
                                    ${StructuredSparsity},
                                    /**/
                                    ${ALayout},
                                    ${BLayout},
                                    ${CLayout},
                                    ${ADataType},
                                    ${BDataType},
                                    ${CDataType},
                                    ${AccDataType},
                                    ${permuteA},
                                    ${permuteB},
                                    ${TransposeC},
                                    ${M},
                                    ${N},
                                    ${K},
                                    ${KBatch},
                                    ${PadM},
                                    ${PadN},
                                    ${PadK}>;
    )__ck__";

    constexpr auto should_pad = [](size_t dim, size_t dim_per_block) {
        return dim % dim_per_block != 0;
    };

    std::vector<GemmSolution> solutions;

    const auto [instance_params, num_instance_params] =
        GetInstanceArray(problem.ADataType, problem.BDataType, problem.CDataType);
    
    for(auto i = 0; i < num_instance_params; ++i) 
    {
        const auto& ip = instance_params[i];

        auto kernel_str = ck::host::InterpolateString(
            template_string,
            {
                /**/
                {"BaseGemmPipeline", PipelineToBaseGemmPipeline(ip.pipeline)},
                {"GemmPipeline", PipelineToGemmPipeline(ip.pipeline)},
                {"DoubleSmemBuffer", ToString(ip.pipeline == Pipeline::V4 ? true : false)},
                {"Scheduler", ToString(ip.scheduler)},
                {"EpilogueSelector", ToString(ip.epilogue)},
                //
                {"TileM", std::to_string(ip.tileM)},
                {"TileN", std::to_string(ip.tileN)},
                {"TileK", std::to_string(ip.tileK)},
                //
                {"WarpM", std::to_string(ip.warpM)},
                {"WarpN", std::to_string(ip.warpN)},
                {"WarpK", std::to_string(ip.warpK)},
                //
                {"WarpTileM", std::to_string(ip.warpTileM)},
                {"WarpTileN", std::to_string(ip.warpTileN)},
                {"WarpTileK", std::to_string(ip.warpTileK)},
                //
                {"StructuredSparsity", ToString(false)},
                //
                {"ALayout", ToString(problem.ALayout)},
                {"BLayout", ToString(problem.BLayout)},
                {"CLayout", ToString(problem.CLayout)},
                //
                {"ADataType", ToString(problem.ADataType)},
                {"BDataType", ToString(problem.BDataType)},
                {"CDataType", ToString(problem.CDataType)},
                {"AccDataType", ToString(problem.AccDataType)},
                //
                {"permuteA", ToString(problem.TransA)},
                {"permuteB", ToString(problem.TransB)},
                {"TransposeC", ToString(problem.TransC)},
                //
                {"M", std::to_string(problem.M)},
                {"N", std::to_string(problem.N)},
                {"K", std::to_string(problem.K)},
                {"KBatch", std::to_string(1)},
                //
                {"PadM", ToString(should_pad(problem.M, ip.tileM))},
                {"PadN", ToString(should_pad(problem.N, ip.tileN))},
                {"PadK", ToString(should_pad(problem.K, ip.tileK))}
                /**/
            });
        solutions.push_back({kernel_str, ip});
    }

    return solutions;
}

const std::string info_string = R"__ck__(
#include <hip/hip_runtime.h>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/gemm/kernel/mgx_prototype_gemm_kernel.hpp>

extern "C" __global__ void info_kernel(dim3* dims) {
    using Kernel = ${KernelInstance}
    constexpr int M = ${M};
    constexpr int N = ${N};
    constexpr int K = ${K};
    constexpr int KBatch = ${KBatch};

    constexpr ck_tile::GemmKernelArgs kernel_args{M, N, K, K, N, {}, N, KBatch};
    static_assert(Kernel::IsSupportedArgument(kernel_args), "Invalid gemm");

    dims[0] = Kernel::GridSize();
    dims[1] = Kernel::BlockSize();
}

)__ck__";

std::tuple<dim3, dim3> get_launch_dims(const std::string& template_str, int M, int N, int K)
{
    auto srcs     = get_headers_for_test();
    auto main_src = ck::host::InterpolateString(info_string,
                                                {{"KernelInstance", template_str},
                                                 {"M", std::to_string(M)},
                                                 {"N", std::to_string(N)},
                                                 {"K", std::to_string(K)},
                                                 {"KBatch", std::to_string(1)}});

    srcs.push_back({"main.cpp", main_src});
    rtc::compile_options opts;
    opts.kernel_name = "info_kernel";
    auto k           = rtc::compile_kernel(srcs, opts);

    rtc::buffer<dim3> dims(2);
    auto dims_gpu = to_gpu(dims);
    k.launch(nullptr, 1, 1)(dims_gpu.data());
    dims = rtc::from_gpu(dims_gpu);
    dims[0].x *= dims[1].x;
    dims[0].y *= dims[1].y;
    dims[0].z *= dims[1].z;

    return {dims[0], dims[1]};
}

const std::string rtc_string = R"__ck__(
#include <ck_tile/ops/gemm/kernel/mgx_prototype_gemm_kernel.hpp>

extern "C" __global__ void f(const ck_tile::half_t* a, const ck_tile::half_t* b, ck_tile::half_t* c) {
    using Kernel = ${KernelInstance}

    constexpr int M = ${M};
    constexpr int N = ${N};
    constexpr int K = ${K};
    constexpr int KBatch = ${KBatch};

    constexpr ck_tile::GemmKernelArgs kernel_args{M, N, K, K, N, {}, N, KBatch};

    static_assert(Kernel::IsSupportedArgument(kernel_args), "Invalid gemm");

    auto run_args = kernel_args;
    run_args.a_ptr = a;
    run_args.b_ptr = b;
    run_args.e_ptr = c;
    Kernel::Run(run_args);
}

)__ck__";

TEST_CASE(prototype)
{
    GemmProblem problem{.M           = 100,
                        .N           = 72,
                        .K           = 144,
                        .ALayout     = Layout::RowMajor,
                        .BLayout     = Layout::RowMajor,
                        .CLayout     = Layout::RowMajor,
                        .ADataType   = DataType::Half,
                        .BDataType   = DataType::Half,
                        .CDataType   = DataType::Half,
                        .AccDataType = DataType::Float,
                        .TransA      = false,
                        .TransB      = false,
                        .TransC      = false};

    const auto solutions = GetGemmSolutions(problem);
    rtc::buffer<half> a(problem.M * problem.K);
    rtc::buffer<half> b(problem.K * problem.N);
    rtc::buffer<half> c(problem.M * problem.N);

    constexpr auto set_buffer = [](rtc::buffer<half>& buff, float val) {
        for(auto i = 0; i < buff.size(); ++i)
        {
            buff[i] = static_cast<half>(val);
        }
    };
    set_buffer(a, 1.0f);
    set_buffer(b, 1.0f);
    auto a_gpu = to_gpu(a);
    auto b_gpu = to_gpu(b);

    int num_invalid = 0;
    int num_failed  = 0;
    for(auto i = 0u; i < solutions.size(); ++i)
    {
        std::cout << "Trying solution " << i << std::endl;
        const auto gemm_kernel_str = solutions[i].kernel;
        dim3 global_work_dims, local_work_dims;

        try
        {
            auto ret         = get_launch_dims(gemm_kernel_str, problem.M, problem.N, problem.K);
            global_work_dims = std::get<0>(ret);
            local_work_dims  = std::get<1>(ret);
            std::cout << "Got launch dims" << std::endl;
        }
        catch(...)
        {
            std::cout << "Invalid kernel instance" << std::endl;
            ++num_invalid;
            continue;
        }

        auto srcs     = get_headers_for_test();
        auto main_src = ck::host::InterpolateString(rtc_string,
                                                    {{"KernelInstance", gemm_kernel_str},
                                                     {"M", std::to_string(problem.M)},
                                                     {"N", std::to_string(problem.N)},
                                                     {"K", std::to_string(problem.K)},
                                                     {"KBatch", std::to_string(1)}});
        srcs.push_back({"main.cpp", main_src});
        rtc::compile_options opts;
        opts.kernel_name = "f";
        auto k           = rtc::compile_kernel(srcs, opts);

        set_buffer(c, 0.0f);
        auto c_gpu = to_gpu(c);
        k.launch(nullptr, global_work_dims, local_work_dims)(
            a_gpu.data(), b_gpu.data(), c_gpu.data());
        std::cout << "Executed kernel" << std::endl;
        c = rtc::from_gpu(c_gpu);

        for(auto i = 0; i < c.size(); ++i)
        {
            if(static_cast<float>(c[i]) != problem.K)
            {
                std::cout << "ERROR: Mismatch on index " << i << ": " << static_cast<float>(c[i])
                          << std::endl;
                ++num_failed;
                break;
            }
        }
    }
    std::cout << "Number of invalid kernels: " << num_invalid << std::endl;
    std::cout << "Number of failed kernels: " << num_failed << std::endl;
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }