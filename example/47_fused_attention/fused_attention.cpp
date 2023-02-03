// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_bias_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

using AElementOp    = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using D0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using B1ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CElementOp    = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
constexpr static auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;
static constexpr auto TensorDefault = ck::tensor_operation::device::TensorSpecialization::Default;

using F16         = ck::half_t;
using F32         = float;
using ADataType   = ck::half_t;
using B0DataType  = ck::half_t;
using B1DataType  = ck::half_t;
using CDataType   = ck::half_t;
using D0DataType  = ck::half_t;
using AccDataType = float;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;

using DeviceOpInstance =
    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<NumDimG,
                                                                                   NumDimM,
                                                                                   NumDimN,
                                                                                   NumDimK,
                                                                                   NumDimO,
                                                                                   F16,
                                                                                   F16,
                                                                                   F16,
                                                                                   F16,
                                                                                   ck::Tuple<F16>,
                                                                                   ck::Tuple<>,
                                                                                   F32,
                                                                                   F16,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   Scale,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   GemmDefault,
                                                                                   TensorDefault,
                                                                                   TensorDefault,
                                                                                   TensorDefault,
                                                                                   TensorDefault,
                                                                                   1,
                                                                                   256,
                                                                                   256,
                                                                                   128,
                                                                                   32,
                                                                                   64,
                                                                                   32,
                                                                                   8,
                                                                                   8,
                                                                                   2,
                                                                                   32,
                                                                                   32,
                                                                                   2,
                                                                                   4,
                                                                                   2,
                                                                                   S<4, 64, 1>,
                                                                                   S<1, 0, 2>,
                                                                                   S<1, 0, 2>,
                                                                                   2,
                                                                                   8,
                                                                                   8,
                                                                                   true,
                                                                                   S<4, 64, 1>,
                                                                                   S<1, 0, 2>,
                                                                                   S<1, 0, 2>,
                                                                                   2,
                                                                                   8,
                                                                                   8,
                                                                                   true,
                                                                                   S<16, 16, 1>,
                                                                                   S<0, 2, 1>,
                                                                                   S<0, 2, 1>,
                                                                                   1,
                                                                                   4,
                                                                                   2,
                                                                                   false,
                                                                                   1,
                                                                                   2,
                                                                                   S<1, 32, 1, 8>,
                                                                                   8,
                                                                                   MaskingSpec>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    int G0      = 48;
    int G1      = 16;
    int M       = 1024;
    int N       = 1024;
    int K       = 64;
    int O       = 64;
    float alpha = 1 / sqrtf(K);
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
    else if(argc == 11)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M  = std::stoi(argv[4]);
        N  = std::stoi(argv[5]);
        K  = std::stoi(argv[6]);
        O  = std::stoi(argv[7]);
        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);

        alpha = std::stof(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        exit(0);
    }
    // A layout [G0, M, G1, K]
    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides{M * G1 * K, K, G1 * K, 1};

    // B0 layout [G0, N, G1, K]
    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{N * G1 * K, K, G1 * K, 1};

    // B1 layout [G0, N, G1, O]
    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides{N * G1 * O, O, 1, G1 * O};

    // C layout [G0, M, G1, O]
    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides{M * G1 * O, O, G1 * O, 1};

    // D layout [G0, M, G1, N]
    std::vector<ck::index_t> d0_gs_ms_os_lengths{G0, G1, M, N};
    std::vector<ck::index_t> d0_gs_ms_os_strides{M * G1 * O, O, G1 * O, 1};

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b0_gs_ns_ks: " << b0_gs_ns_ks.mDesc << std::endl;
    std::cout << "b1_gs_os_ns: " << b1_gs_os_ns.mDesc << std::endl;
    std::cout << "c_gs_ms_os: " << c_gs_ms_os_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<2>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * G0 * G1 * M * K);
    DeviceMem b0_device_buf(sizeof(B0DataType) * G0 * G1 * N * K);
    DeviceMem d0_device_buf(sizeof(D0DataType) * G0 * G1 * M * N);
    DeviceMem b1_device_buf(sizeof(B1DataType) * G0 * G1 * O * N);
    DeviceMem c_device_buf(sizeof(CDataType) * G0 * G1 * M * O);

    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();

    auto argument = device_op.MakeArgument(
        static_cast<const ADataType*>(a_device_buf.GetDeviceBuffer()),
        static_cast<const B0DataType*>(b0_device_buf.GetDeviceBuffer()),
        static_cast<const B1DataType*>(b1_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
        std::array<void*, 1>{d0_device_buf.GetDeviceBuffer()}, // p_acc0_biases
        {},                                                    // p_acc1_biases
        a_gs_ms_ks_lengths,
        a_gs_ms_ks_strides,
        b0_gs_ns_ks_lengths,
        b0_gs_ns_ks_strides,
        b1_gs_os_ns_lengths,
        b1_gs_os_ns_strides,
        c_gs_ms_os_lengths,
        c_gs_ms_os_strides,
        std::array<std::vector<ck::index_t>, 1>{
            d0_gs_ms_os_lengths}, // acc0_biases_gs_ms_ns_lengths
        std::array<std::vector<ck::index_t>, 1>{
            d0_gs_ms_os_strides}, // acc0_biases_gs_ms_ns_strides
        {},                       // acc1_biases_gs_ms_os_lengths
        {},                       // acc1_biases_gs_ms_os_strides
        AElementOp{},
        B0ElementOp{},
        Acc0ElementOp{alpha},
        B1ElementOp{},
        CElementOp{});

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * G0 * G1;
    std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                             sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                            G0 * G1;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    if(do_verification) {}

    return 0;
}
