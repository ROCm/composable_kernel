// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/host_tensor/host_common_util.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType         = F16;
using BDataType         = F16;
using CDataType         = F16;
using DDataType         = F16;
using ReduceAccDataType = F32;
using GemmAccDataType   = F32;
using CShuffleDataType  = F32;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp                   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp                   = ck::tensor_operation::element_wise::PassThrough;
using DElementOp                   = ck::tensor_operation::element_wise::PassThrough;
static constexpr auto DGlobalMemOp = ck::InMemoryDataOperationEnum::Set;

static constexpr auto GemmSpecialization =
    ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGroupedGemmSoftmaxInstance = ck::tensor_operation::device::DeviceGroupedGemmSoftmax_Xdl_CShuffle
    <Row,                        // typename ALayout
     Col,                        // typename BLayout
     Row,                        // typename CLayout
     F16,                        // typename ADataType
     F16,                        // typename BDataType
     F16,                        // typename DDataType
     F32,                        // typename GemmAccDataType
     F32,                        // typename CShuffleDataType
     F32,                        // typename ReduceAccDataType
     AElementOp,                 // typename AElementwiseOperation
     BElementOp,                 // typename BElementwiseOperation
     DElementOp,                 // typename DElementwiseOperation
     DGlobalMemOp,               // typename DGlobalMemoryDataOperation
     GemmSpecialization,         // GemmSpecialization GemmSpecialization
     1,                          // index_t NumPrefetch
     256,                        // index_t BlockSize
     64,                         // index_t MPerBlock
     512,                        // index_t NPerBlock
     32,                         // index_t KPerBlock
     8,                          // index_t AK1
     8,                          // index_t BK1
     32,                         // index_t MPerXDL
     32,                         // index_t NPerXDL
     2,                          // index_t MXdlPerWave
     4,                          // index_t NXdlPerWave
     S<4, 64, 1>,                // typename ABlockTransferThreadClusterLengths_AK0_M_AK1
     S<1,  0, 2>,                // typename ABlockTransferThreadClusterArrangeOrder
     S<1,  0, 2>,                // typename ABlockTransferSrcAccessOrder
     2,                          // index_t ABlockTransferSrcVectorDim
     8,                          // index_t ABlockTransferSrcScalarPerVector
     8,                          // index_t ABlockTransferDstScalarPerVector_AK1
     1,                          // index_t ABlockLdsExtraM
     S<4, 64, 1>,                // typename BBlockTransferThreadClusterLengths_BK0_N_BK1
     S<1, 0, 2>,                 // typename BBlockTransferThreadClusterArrangeOrder
     S<1, 0, 2>,                 // typename BBlockTransferSrcAccessOrder
     2,                          // index_t BBlockTransferSrcVectorDim
     8,                          // index_t BBlockTransferSrcScalarPerVector
     8,                          // index_t BBlockTransferDstScalarPerVector_BK1
     1,                          // index_t BBlockLdsExtraN
     1,                          // index_t CShuffleMXdlPerWavePerShuffle
     4,                          // index_t CShuffleNXdlPerWavePerShuffle
     8,                          // index_t MThreadClusterSize
     32,                         // index_t NThreadClusterSize
     4,                          // index_t MThreadSliceSize
     16,                         // index_t NThreadSliceSize
     1,                          // index_t InSrcVectorDim
     8,                          // index_t InSrcVectorSize
     8>;                         // index_t OutDstVectorSize
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        GemmAccDataType,
                                                                        GemmAccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        DElementOp>;

using ReferenceInstance =
    ck::tensor_operation::host::ReferenceSoftmax<CShuffleDataType, DDataType, ReduceAccDataType>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    int group_count = rand() % 16 + 1;

    ReduceAccDataType alpha = 1.0;
    ReduceAccDataType beta  = 0.0;
    const std::vector<int> reduceDims{1};

    if(argc == 1)
    {
        // do nothing
    }
    else if(argc == 5)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        group_count     = std::stoi(argv[4]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4: group count (greater than or equal to 1)\n");
        exit(0);
    }

    // int group_count = rand() % 16 + 1;

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmDesc> gemm_shapes;
    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_d;

    gemm_shapes.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        int M = 256 + 256 * i;
        int N = 512;
        // int K = 64;
        int K = 64 + 64 * i;

        gemm_shapes.push_back({M, N, K, K, K, N});
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

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<GemmAccDataType>> c_host_tensors;
    std::vector<Tensor<DDataType>> d_host_tensors;
    std::vector<Tensor<DDataType>> d_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    c_host_tensors.reserve(group_count);
    d_host_tensors.reserve(group_count);
    d_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, d_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    d_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(std::size_t i = 0; i < gemm_shapes.size(); i++)
    {
        a_tensors.push_back(Tensor<ADataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M_, gemm_shapes[i].K_, gemm_shapes[i].stride_A_, ALayout{})));
        b_tensors.push_back(Tensor<BDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].K_, gemm_shapes[i].N_, gemm_shapes[i].stride_B_, BLayout{})));
        c_host_tensors.push_back(Tensor<GemmAccDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M_, gemm_shapes[i].N_, gemm_shapes[i].stride_C_, CLayout{})));
        d_host_tensors.push_back(Tensor<DDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M_, gemm_shapes[i].N_, gemm_shapes[i].stride_C_, CLayout{})));
        d_device_tensors.push_back(Tensor<DDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M_, gemm_shapes[i].N_, gemm_shapes[i].stride_C_, CLayout{})));

        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_k_n: " << b_tensors[i].mDesc << " c_m_n: " << c_host_tensors[i].mDesc
                  << " d_m_n: " << d_device_tensors[i].mDesc << std::endl;

        flop += std::size_t(2) * gemm_shapes[i].M_ * gemm_shapes[i].K_ * gemm_shapes[i].N_;
        num_btype += sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize() +
                     sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(GemmAccDataType) * c_host_tensors[i].mDesc.GetElementSize() +
                     sizeof(DDataType) * d_device_tensors[i].mDesc.GetElementSize();

        switch(init_method)
        {
        case 0: break;
        case 1:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
            break;
        case 2:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
            break;
        default:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<0>{});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        }
    }

    for(std::size_t i = 0; i < gemm_shapes.size(); i++)
    {
        a_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ADataType) * a_tensors[i].mDesc.GetElementSpace()));
        b_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(BDataType) * b_tensors[i].mDesc.GetElementSpace()));
        d_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(DDataType) * d_device_tensors[i].mDesc.GetElementSpace()));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_d.push_back(d_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = DElementOp{};
    auto d_element_op = DElementOp{};

    auto gemm    = DeviceGroupedGemmSoftmaxInstance{};
    auto invoker = gemm.MakeInvoker();

    // do GEMM
    auto argument = gemm.MakeArgument(
        p_a, p_b, p_d, gemm_shapes, a_element_op, b_element_op, d_element_op, alpha);

    DeviceMem gemm_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, gemm_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;

    int noo_zero_count_device = 0;
    int noo_zero_count_host   = 0;
    if(do_verification)
    {
        for(std::size_t i = 0; i < gemm_shapes.size(); i++)
        {
            d_tensors_device[i]->FromDevice(d_device_tensors[i].mData.data());
            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                      b_tensors[i],
                                                      c_host_tensors[i],
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op);

            ref_invoker.Run(ref_argument);

            ReferenceInstance ref;
            auto ref_arg =
                ref.MakeArgument(c_host_tensors[i], d_host_tensors[i], alpha, beta, reduceDims);
            auto invoker = ref.MakeInvoker();
            invoker.Run(ref_arg);

            // pass = ck::utils::check_err(d_device_tensors[i].mData,
            //                         d_host_tensors[i].mData,
            //                         "Error: Incorrect results! D",
            //                         1e-4,
            //                         1e-5);

            pass = ck::utils::check_err(
                d_device_tensors[i].mData, d_host_tensors[i].mData, "Error: Incorrect results! D");
        }
    }

    if(pass)
    {
        std::cout << "Test Pass!!!!!!!!!!!!!!!!" << std::endl;
    }
    else
    {
        std::cout << "Test Fail................" << std::endl;
    }

    return pass ? 0 : 1;
}
