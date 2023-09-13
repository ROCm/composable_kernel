
#pragma once

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp"

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = Col;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmDl<
            ck::half_t,
            ck::half_t,
            ck::half_t,
            float,
            ck::tensor_layout::gemm::ColumnMajor,
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::device::GemmSpecialization::Default,
            256,
            128,
            128,
            16,
            2,
            4,
            4,
            1,
            S<8, 2>,
            S<8, 2>,
            S<2, 1, 4, 2>,
            S<8, 1,  32, 1>,
            S<0, 3, 1, 2>,
            S<0, 3, 1, 2>,
            S<1, 1, 4, 1>,
            S<0, 3, 1, 2>,
            S<1, 1, 4, 2>,
            S<2, 1, 4, 2>,
            S<8, 1, 32, 1>,
            S<0, 3, 1, 2>,
            S<0, 3, 1, 2>,
            S<1, 1, 4, 1>,
            S<0, 3, 1, 2>,
            S<1, 1, 4, 2>,
            S<0, 1, 2, 3, 4, 5>,
            5,
            4>;

    using ReferenceGemmInstance = ck::tensor_operation::host::
        ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;


bool run_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto& [M, N, K, StrideA, StrideB, StrideC] = problem_size;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ck::half_t> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ck::tensor_layout::gemm::ColumnMajor{}));
    Tensor<ck::half_t> b_k_n(f_host_tensor_descriptor(K, N, StrideB, ck::tensor_layout::gemm::RowMajor{}));

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ck::half_t>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<ck::half_t>{-5.f, 5.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ck::half_t>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<ck::half_t>{-1.f, 1.f}(b_k_n);
    }

    Tensor<ck::half_t> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<ck::half_t> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(ck::half_t) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(ck::half_t) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(ck::half_t) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());


    auto a_element_op = ck::tensor_operation::element_wise::PassThrough{};
    auto b_element_op = ck::tensor_operation::element_wise::PassThrough{};
    auto c_element_op = ck::tensor_operation::element_wise::PassThrough{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(

        static_cast<ck::half_t*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<ck::half_t*>(b_k_n_device_buf.GetDeviceBuffer()),
        static_cast<ck::half_t*>(c_m_n_device_buf.GetDeviceBuffer()),
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideC,
        a_element_op,
        b_element_op,
        c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop = 2_uz * M * N * K;
    std::size_t num_btype =
        sizeof(ck::half_t) * M * K + sizeof(ck::half_t) * K * N + sizeof(ck::half_t) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        return ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
    }

    return true;
}

bool run_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
