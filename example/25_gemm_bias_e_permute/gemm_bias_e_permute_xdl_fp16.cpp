// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_bias_e_permute_xdl.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F16;
using EDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using DLayout = Row;
using ELayout = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmBiasEPermute_Xdl
//######| ALayout| BLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
//######|        |        |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
//######|        |        |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
//######|        |        |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        < ALayout, BLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType,  DDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>;
// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::index_t M0 = 4;
    ck::index_t M1 = 32;
    ck::index_t M2 = 128;
    ck::index_t N0 = 16;
    ck::index_t N1 = 256;

    // GEMM shape
    ck::index_t M = M0 * M1 * M2;
    ck::index_t N = N0 * N1;
    ck::index_t K = 128;

    ck::index_t stride_A = K;
    ck::index_t stride_B = K;

#if 1
    // E = [M0, N0, M1, N1, M2]
    ck::index_t stride_E_M0 = N0 * M1 * N1 * M2;
    ck::index_t stride_E_M1 = N1 * M2;
    ck::index_t stride_E_M2 = 1;
    ck::index_t stride_E_N0 = M1 * N1 * M2;
    ck::index_t stride_E_N1 = M2;

    // D = [0, N0, 0, N1, 0]
    ck::index_t stride_D_M0 = 0;
    ck::index_t stride_D_M1 = 0;
    ck::index_t stride_D_M2 = 0;
    ck::index_t stride_D_N0 = N1;
    ck::index_t stride_D_N1 = 1;
#else
    // D = [0, 0, 0, N0, N1]
    ck::index_t stride_D_M0 = 0;
    ck::index_t stride_D_M1 = 0;
    ck::index_t stride_D_M2 = 0;
    ck::index_t stride_D_N0 = N1;
    ck::index_t stride_D_N1 = 1;

    // E = [M0, M1, M2, N0, N1]
    ck::index_t stride_E_M0 = M1 * M2 * N0 * N1;
    ck::index_t stride_E_M1 = M2 * N0 * N1;
    ck::index_t stride_E_M2 = N0 * N1;
    ck::index_t stride_E_N0 = N1;
    ck::index_t stride_E_N1 = 1;
#endif

    const ck::tensor_operation::device::DEGridDesc_M0_M1_M2_N0_N1 d_grid_desc{
        M0, M1, M2, N0, N1, stride_D_M0, stride_D_M1, stride_D_M2, stride_D_N0, stride_D_N1};
    const ck::tensor_operation::device::DEGridDesc_M0_M1_M2_N0_N1 e_grid_desc{
        M0, M1, M2, N0, N1, stride_E_M0, stride_E_M1, stride_E_M2, stride_E_N0, stride_E_N1};

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
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

    auto f_host_de_tensor_descriptor =
        [](ck::tensor_operation::device::DEGridDesc_M0_M1_M2_N0_N1 de_grid_desc) {
            std::size_t m0        = de_grid_desc.M0_;
            std::size_t m1        = de_grid_desc.M1_;
            std::size_t m2        = de_grid_desc.M2_;
            std::size_t n0        = de_grid_desc.N0_;
            std::size_t n1        = de_grid_desc.N1_;
            std::size_t stride_m0 = de_grid_desc.stride_M0_;
            std::size_t stride_m1 = de_grid_desc.stride_M1_;
            std::size_t stride_m2 = de_grid_desc.stride_M2_;
            std::size_t stride_n0 = de_grid_desc.stride_N0_;
            std::size_t stride_n1 = de_grid_desc.stride_N1_;
            return HostTensorDescriptor(
                std::vector<std::size_t>({m0, m1, m2, n0, n1}),
                std::vector<std::size_t>({stride_m0, stride_m1, stride_m2, stride_n0, stride_n1}));
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, stride_A, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, stride_B, BLayout{}));
    Tensor<DDataType> d_m0_m1_m2_n0_n1(f_host_de_tensor_descriptor(d_grid_desc));
    Tensor<EDataType> e_m0_m1_m2_n0_n1_host_result(f_host_de_tensor_descriptor(e_grid_desc));
    Tensor<EDataType> e_m0_m1_m2_n0_n1_device_result(f_host_de_tensor_descriptor(e_grid_desc));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "d_m0_m1_m2_n0_n1: " << d_m0_m1_m2_n0_n1.mDesc << std::endl;
    std::cout << "e_m0_m1_m2_n0_n1: " << e_m0_m1_m2_n0_n1_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_m0_m1_m2_n0_n1.GenerateTensorValue(GeneratorTensor_2<DDataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_m0_m1_m2_n0_n1.GenerateTensorValue(GeneratorTensor_3<DDataType>{0.0, 1.0});
    }

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d_m0_m1_m2_n0_n1_device_buf(sizeof(DDataType) *
                                          d_m0_m1_m2_n0_n1.mDesc.GetElementSpaceSize());
    DeviceMem e_m0_m1_m2_n0_n1_device_buf(
        sizeof(EDataType) * e_m0_m1_m2_n0_n1_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
    d_m0_m1_m2_n0_n1_device_buf.ToDevice(d_m0_m1_m2_n0_n1.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(a_m_k_device_buf.GetDeviceBuffer(),
                                           b_k_n_device_buf.GetDeviceBuffer(),
                                           d_m0_m1_m2_n0_n1_device_buf.GetDeviceBuffer(),
                                           e_m0_m1_m2_n0_n1_device_buf.GetDeviceBuffer(),
                                           M,
                                           N,
                                           K,
                                           stride_A,
                                           stride_B,
                                           d_grid_desc,
                                           e_grid_desc,
                                           a_element_op,
                                           b_element_op,
                                           cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = std::size_t(2) * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                            sizeof(DDataType) * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << device_op.GetTypeString() << std::endl;

    if(do_verification)
    {
        Tensor<AccDataType> c_m_n(HostTensorDescriptor(
            std::vector<std::size_t>{static_cast<std::size_t>(M), static_cast<std::size_t>(N)}));

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                PassThrough>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m0 = 0; m0 < M0; ++m0)
            for(int m1 = 0; m1 < M1; ++m1)
                for(int m2 = 0; m2 < M2; ++m2)
                    for(int n0 = 0; n0 < N0; ++n0)
                        for(int n1 = 0; n1 < N1; ++n1)
                        {
                            int m = m0 * M1 * M2 + m1 * M2 + m2;
                            int n = n0 * N1 + n1;

                            cde_element_op(e_m0_m1_m2_n0_n1_host_result(m0, m1, m2, n0, n1),
                                           ck::type_convert<EDataType>(c_m_n(m, n)),
                                           d_m0_m1_m2_n0_n1(m0, m1, m2, n0, n1));
                        }

        e_m0_m1_m2_n0_n1_device_buf.FromDevice(e_m0_m1_m2_n0_n1_device_result.mData.data());

        return ck::utils::check_err(e_m0_m1_m2_n0_n1_device_result.mData,
                                    e_m0_m1_m2_n0_n1_host_result.mData)
                   ? 0
                   : 1;
    }

    return 0;
}
