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
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_gemm_xdl_c_shuffle_bias_2d.hpp"
#include "element_wise_operation.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::Add;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmXdl_C_Shuffle_Bias_2d<
    ADataType,              // ADataType
    BDataType,              // BDataType
    CDataType,              // CDataType
    AccDataType,            // AccDataType
    ALayout,                // ALayout
    BLayout,                // BLayout
    CLayout,                // CLayout
    AElementOp,             // AElementwiseOperation
    BElementOp,             // BElementwiseOperation
    CElementOp,             // CElementwiseOperation
    256,                    // BlockSize
    256,                    // MPerBlock
    128,                    // NPerBlock
    4,                      // K0PerBlock
    8,                      // K1
    32,                     // MPerXDL
    32,                     // NPerXDL
    4,                      // MXdlPerWave
    2,                      // NXdlPerWave
    S<4, 64, 1>,            // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,             // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,             // ABlockTransferSrcAccessOrder
    2,                      // ABlockTransferSrcVectorDim
    8,                      // ABlockTransferSrcScalarPerVector
    8,                      // ABlockTransferDstScalarPerVector_K1
    true,                   // ABlockLdsAddExtraM
    S<4, 64, 1>,            // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,             // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,             // BBlockTransferSrcAccessOrder
    2,                      // BBlockTransferSrcVectorDim
    8,                      // BBlockTransferSrcScalarPerVector
    8,                      // BBlockTransferDstScalarPerVector_K1
    true,                   // BBlockLdsAddExtraN
    1,                      // CShuffleMXdlPerWavePerShuffle
    1,                      // CShuffleNXdlPerWavePerShuffle
    S<1, 1, 32, 1, 1, 8>,   // CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
    8>;                     // CBlockTransferScalarPerVector_NWaveNPerXdl
// clang-format on

template <typename AType,
          typename BType,
          typename CType,
          typename C0Type,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
static void host_verify(const Tensor<AType>& a_m_k,
                        const Tensor<BType>& b_k_n,
                        const Tensor<C0Type>& c0_k_n,
                        Tensor<CType>& c_m_n,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CElementwiseOperation& c_element_op)
{
    auto f_mk_kn_mn = [&](auto m, auto n) {
        const int K = a_m_k.mDesc.GetLengths()[1];

        AccDataType v = 0;
        AccDataType a = 0;
        AccDataType b = 0;
        for(int k = 0; k < K; ++k)
        {
            a_element_op(a, a_m_k(m, k));
            b_element_op(b, b_k_n(k, n));
            v += a * b;
        }

        CType y = static_cast<CType>(v);

        c_element_op(c_m_n(m, n), y, c0_k_n(m, n));
    };

    make_ParallelTensorFunctor(f_mk_kn_mn,
                               c_m_n.mDesc.GetLengths()[0],
                               c_m_n.mDesc.GetLengths()[1])(std::thread::hardware_concurrency());
}

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideC = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
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
    Tensor<BDataType> c0_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<BDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<BDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c0_m_n: " << c0_m_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        c0_m_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        c0_m_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c0_m_n_device_buf(sizeof(CDataType) * c0_m_n.mDesc.GetElementSpace());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
    c0_m_n_device_buf.ToDevice(c0_m_n.mData.data());
    c_m_n_device_buf.ToDevice(c_m_n_device_result.mData.data());

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c0_m_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      AElementOp{},
                                      BElementOp{},
                                      CElementOp{});

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, nrepeat);

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

    if(do_verification)
    {
        host_verify(
            a_m_k, b_k_n, c0_m_n, c_m_n_host_result, AElementOp{}, BElementOp{}, CElementOp{});

        check_error(c_m_n_host_result, c_m_n_device_result);
    }
}
