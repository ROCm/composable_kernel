#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "gemm_common.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_gemm_xdl.hpp"

// Currently ADataType and BDataType need to be the same
using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

// NT problem
using ALayout = ck::tensor_layout::RowMajor;
using BLayout = ck::tensor_layout::ColumnMajor;
using CLayout = ck::tensor_layout::RowMajor;

// tuning parameter for NT problem
using DeviceGemm0 = ck::tensor_operation::device::DeviceGemmXdl<
    ADataType,              // ADataType,
    BDataType,              // BDataType,
    CDataType,              // CDataType,
    AccDataType,            // AccDataType,
    ALayout,                // ALayout,
    BLayout,                // BLayout,
    CLayout,                // CLayout,
    256,                    // BlockSize,
    256,                    // MPerBlock,
    128,                    // NPerBlock,
    4,                      // K0PerBlock,
    8,                      // K1,
    32,                     // MPerXDL,
    32,                     // NPerXDL,
    4,                      // MXdlPerWave,
    2,                      // NXdlPerWave,
    ck::Sequence<1, 4, 8>,  // ABlockTransferThreadSliceLengths_K0_M_K1,
    ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1,
    ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder,
    ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder,
    2,                      // ABlockTransferSrcVectorDim,
    8,                      // ABlockTransferSrcScalarPerVector,
    8,                      // ABlockTransferDstScalarPerVector_K1,
    ck::Sequence<1, 2, 8>,  // BBlockTransferThreadSliceLengths_K0_N_K1,
    ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1,
    ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder,
    ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder,
    2,                      // BBlockTransferSrcVectorDim,
    8,                      // BBlockTransferSrcScalarPerVector,
    8,                      // BBlockTransferDstScalarPerVector_K1,
    7,                      // CThreadTransferSrcDstVectorDim,
    1,                      // CThreadTransferDstScalarPerVector,
    true,                   // ABlockLdsAddExtraM,
    true>;                  // BBlockLdsAddExtraN

// tuning parameter for NT problem
using DeviceGemm1 = ck::tensor_operation::device::DeviceGemmXdl<
    ADataType,              // ADataType,
    BDataType,              // BDataType,
    CDataType,              // CDataType,
    AccDataType,            // AccDataType,
    ALayout,                // ALayout,
    BLayout,                // BLayout,
    CLayout,                // CLayout,
    256,                    // BlockSize,
    128,                    // MPerBlock,
    128,                    // NPerBlock,
    4,                      // K0PerBlock,
    8,                      // K1,
    32,                     // MPerXDL,
    32,                     // NPerXDL,
    2,                      // MXdlPerWave,
    2,                      // NXdlPerWave,
    ck::Sequence<1, 2, 8>,  // ABlockTransferThreadSliceLengths_K0_M_K1,
    ck::Sequence<4, 64, 1>, // ABlockTransferThreadClusterLengths_K0_M_K1,
    ck::Sequence<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder,
    ck::Sequence<1, 0, 2>,  // ABlockTransferSrcAccessOrder,
    2,                      // ABlockTransferSrcVectorDim,
    8,                      // ABlockTransferSrcScalarPerVector,
    8,                      // ABlockTransferDstScalarPerVector_K1,
    ck::Sequence<1, 2, 8>,  // BBlockTransferThreadSliceLengths_K0_N_K1,
    ck::Sequence<4, 64, 1>, // BBlockTransferThreadClusterLengths_K0_N_K1,
    ck::Sequence<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder,
    ck::Sequence<1, 0, 2>,  // BBlockTransferSrcAccessOrder,
    2,                      // BBlockTransferSrcVectorDim,
    8,                      // BBlockTransferSrcScalarPerVector,
    8,                      // BBlockTransferDstScalarPerVector_K1,
    7,                      // CThreadTransferSrcDstVectorDim,
    1,                      // CThreadTransferDstScalarPerVector,
    true,                   // ABlockLdsAddExtraM,
    true>;                  // BBlockLdsAddExtraN

int main(int argc, char* argv[])
{
    using namespace ck;

    if(argc != 11)
    {
        printf("arg1 to 4: do_verification, init_method, do_log, nrepeat\n");
        printf("arg5 to 10: M, N, K, StrideA, StrideB, StrideC\n");
        exit(1);
    }

    const bool do_verification = std::stoi(argv[1]);
    const int init_method      = std::stoi(argv[2]);
    const bool do_log          = std::stoi(argv[3]);
    const int nrepeat          = std::stoi(argv[4]);

    const index_t M = std::stoi(argv[5]);
    const index_t N = std::stoi(argv[6]);
    const index_t K = std::stoi(argv[7]);

    const index_t StrideA = std::stoi(argv[8]);
    const index_t StrideB = std::stoi(argv[9]);
    const index_t StrideC = std::stoi(argv[10]);

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(is_same<decltype(layout), tensor_layout::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<BDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0:
        // no initialization
        break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_1{});
        b_k_n.GenerateTensorValue(GeneratorTensor_1{});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1{});
        b_k_n.GenerateTensorValue(GeneratorTensor_2{-5, 5});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_1{});
        break;
    case 4:
        a_m_k.GenerateTensorValue(GeneratorTensor_2{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<float>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<float>{-0.5, 0.5});
    }

    // run on GPU
    {
        DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
        DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
        DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());

        a_m_k_device_buf.ToDevice(a_m_k.mData.data());
        b_k_n_device_buf.ToDevice(b_k_n.mData.data());
        c_m_n_device_buf.ToDevice(c_m_n_device_result.mData.data());

        using BaseOp      = ck::tensor_operation::device::BaseOperator;
        using BaseInvoker = ck::tensor_operation::device::BaseInvoker;
        using BaseArg     = ck::tensor_operation::device::BaseArgument;

        std::vector<std::tuple<std::unique_ptr<BaseOp>,
                               std::unique_ptr<BaseInvoker>,
                               std::unique_ptr<BaseArg>>>
            device_gemm_tuples;

        {
            using Gemm         = DeviceGemm0;
            using GemmInvoker  = Gemm::Invoker;
            using GemmArgument = Gemm::Argument;

            auto gemm    = Gemm{};
            auto invoker = gemm.MakeInvoker();
            auto argument =
                gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                  static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                  static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                  M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  StrideC);

            device_gemm_tuples.push_back(std::make_tuple(std::make_unique<Gemm>(gemm),
                                                         std::make_unique<GemmInvoker>(invoker),
                                                         std::make_unique<GemmArgument>(argument)));
        }

        {
            using Gemm         = DeviceGemm1;
            using GemmInvoker  = Gemm::Invoker;
            using GemmArgument = Gemm::Argument;

            auto gemm    = Gemm{};
            auto invoker = gemm.MakeInvoker();
            auto argument =
                gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                  static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                  static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                  M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  StrideC);

            device_gemm_tuples.push_back(std::make_tuple(std::make_unique<Gemm>(gemm),
                                                         std::make_unique<GemmInvoker>(invoker),
                                                         std::make_unique<GemmArgument>(argument)));
        }

        for(auto& device_gemm_tuple : device_gemm_tuples)
        {
            auto& gemm_ptr     = std::get<0>(device_gemm_tuple);
            auto& invoker_ptr  = std::get<1>(device_gemm_tuple);
            auto& argument_ptr = std::get<2>(device_gemm_tuple);

            if(!gemm_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                throw std::runtime_error(
                    "wrong! device_gemm with the specified compilation parameters does "
                    "not support this GEMM problem");
            }

            // run kernel
            for(int i = 0; i < 5; ++i)
            {
                float ave_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

                float perf = static_cast<float>((std::size_t(2) * M * N * K)) /
                             (std::size_t(1000) * 1000 * 1000) / ave_time;

                std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                          << std::endl;
            }
        }

        // copy result back to host
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());
    }

    if(do_verification)
    {
        host_gemm_mk_kn_mn(a_m_k, b_k_n, c_m_n_host_result);

        check_error(c_m_n_host_result, c_m_n_device_result);

        if(do_log)
        {
            LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                << std::endl;
            LogRangeAsType<float>(std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                << std::endl;
        }
    }
}
