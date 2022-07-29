// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_reduce_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType         = F16;
using BDataType         = F16;
using CDataType         = F16;
using ReduceAccDataType = F32;
using DDataType         = F32;
using DPtrsGlobal       = ck::Tuple<DDataType*>;
using AccDataType       = float;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp  = ck::tensor_operation::element_wise::PassThrough;
using BElementOp  = ck::tensor_operation::element_wise::PassThrough;
using CElementOp  = ck::tensor_operation::element_wise::PassThrough;
using DReduceOp   = ck::reduce::Max;
using DxsReduceOp = ck::Tuple<DReduceOp>;

using UnaryIdenticElementOp = ck::tensor_operation::element_wise::PassThrough;
using DxsInElementOp        = ck::Tuple<UnaryIdenticElementOp>;
using DxsOutElementOp       = ck::Tuple<UnaryIdenticElementOp>;

using DGlobalMemOp =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicMax>;

static constexpr auto GemmSpecialization =
    ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGroupedGemmReduceInstance = ck::tensor_operation::device::DeviceGroupedGemmReduce_Xdl_CShuffle
//######| ALayout| BLayout| CLayout|AData| BData| CData|  GemmAcc| CShuffle| ReduceAcc|         DData|           A|           B|           C|         Dxs|     DxsInEleOp|     DxsAccEleOp|            D|               GEMM| NumPrefetch| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|              CReduce| CReduceThreadLds2VGprCopy| CReduceThreadVgpr2GlobalCopy|
//######|        |        |        | Type|  Type|  Type| DataType| DataType|  DataType|    Type Tuple| Elementwise| Elementwise| Elementwise|      Reduce|               |                |   MemoryData|     Spacialization|            |  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|            _MBlock_MPerBlock| ScalarPerVector| ThreadClusterLengths|     SrcDstScalarPerVector|        SrcDstScalarPerVector|
//######|        |        |        |     |      |      |         |         |          |              |   Operation|   Operation|   Operation|   Operation|               |                |    Operation|                   |            |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|            _NBlock_NPerBlock|      _NPerBlock| _MPerBlock_NPerBlock|                _NPerBlock|                   _MPerBlock|
//######|        |        |        |     |      |      |         |         |          |              |            |            |            |            |               |                |             |                   |            |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                     |                          |                             |
       <     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,       F32,   DPtrsGlobal,  AElementOp,  BElementOp,  CElementOp, DxsReduceOp, DxsInElementOp, DxsOutElementOp, DGlobalMemOp, GemmSpecialization,           1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8,             S<64, 4>,                         4,                            1>;
//         <     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,       F32,   DPtrsGlobal,  AElementOp,  BElementOp,  CElementOp, DxsReduceOp, DxsInElementOp, DxsOutElementOp, DGlobalMemOp, GemmSpecialization,           1,   256,    64,   512,    32,   8,   8,   32,   32,    2,    4,     S<4,  64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           4,               S<1, 32, 1, 8>,               8,             S<32, 8>,                         4,                            1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    int group_count = rand() % 16 + 1;

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
    std::vector<void*> p_c;
    std::vector<DPtrsGlobal> dxs_global;

    gemm_shapes.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        int M = 256 + 256 * i;
        int N = 128 + 128 * i;
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
    std::vector<Tensor<CDataType>> c_host_tensors;
    std::vector<Tensor<CDataType>> c_device_tensors;
    std::vector<Tensor<DDataType>> d_host_tensors;
    std::vector<Tensor<DDataType>> d_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    c_host_tensors.reserve(group_count);
    c_device_tensors.reserve(group_count);
    d_host_tensors.reserve(group_count);
    d_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, c_tensors_device,
        d_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);
    d_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(std::size_t i = 0; i < gemm_shapes.size(); i++)
    {
        a_tensors.push_back(Tensor<ADataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M_, gemm_shapes[i].K_, gemm_shapes[i].stride_A_, ALayout{})));
        b_tensors.push_back(Tensor<BDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].K_, gemm_shapes[i].N_, gemm_shapes[i].stride_B_, BLayout{})));
        c_host_tensors.push_back(Tensor<CDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M_, gemm_shapes[i].N_, gemm_shapes[i].stride_C_, CLayout{})));
        c_device_tensors.push_back(Tensor<CDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M_, gemm_shapes[i].N_, gemm_shapes[i].stride_C_, CLayout{})));
        d_host_tensors.push_back(Tensor<DDataType>(HostTensorDescriptor(
            std::vector<std::size_t>({static_cast<std::size_t>(gemm_shapes[i].M_)}))));
        d_device_tensors.push_back(Tensor<DDataType>(HostTensorDescriptor(
            std::vector<std::size_t>({static_cast<std::size_t>(gemm_shapes[i].M_)}))));

        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_k_n: " << b_tensors[i].mDesc << " c_m_n: " << c_device_tensors[i].mDesc
                  << " d_m: " << d_device_tensors[i].mDesc << std::endl;

        flop += std::size_t(2) * gemm_shapes[i].M_ * gemm_shapes[i].K_ * gemm_shapes[i].N_;
        num_btype += sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize() +
                     sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(CDataType) * c_device_tensors[i].mDesc.GetElementSize();

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
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * c_device_tensors[i].mDesc.GetElementSpace()));
        d_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(DDataType) * d_device_tensors[i].mDesc.GetElementSpace()));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
        dxs_global.push_back(
            ck::make_tuple(static_cast<DDataType*>(d_tensors_device[i]->GetDeviceBuffer())));
    }

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    auto gemm    = DeviceGroupedGemmReduceInstance{};
    auto invoker = gemm.MakeInvoker();

    // do GEMM
    auto argument = gemm.MakeArgument(p_a,
                                      p_b,
                                      p_c,
                                      dxs_global,
                                      gemm_shapes,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op,
                                      DxsInElementOp{},
                                      DxsOutElementOp{});

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

    if(do_verification)
    {
        for(std::size_t i = 0; i < gemm_shapes.size(); i++)
        {
            c_tensors_device[i]->FromDevice(c_device_tensors[i].mData.data());
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

            auto d_reduce_op = DReduceOp{};

            int M = ck::type_convert<int>(gemm_shapes[i].M_);
            int N = ck::type_convert<int>(gemm_shapes[i].N_);

            for(int m = 0; m < M; ++m)
            {
                auto d_acc = d_reduce_op.GetIdentityValue<ReduceAccDataType>();

                for(int n = 0; n < N; ++n)
                {
                    auto c_val = ck::type_convert<ReduceAccDataType>(c_host_tensors[i](m, n));
                    ReduceAccDataType d_val = 0;

                    UnaryIdenticElementOp{}(d_val, c_val);
                    d_reduce_op(d_acc, d_val);
                }

                d_host_tensors[i](m) = ck::type_convert<DDataType>(d_acc);
            }

            pass = ck::utils::check_err(c_device_tensors[i].mData,
                                        c_host_tensors[i].mData,
                                        "Error: Incorrect results c") &&
                   ck::utils::check_err(d_device_tensors[i].mData,
                                        d_host_tensors[i].mData,
                                        "Error: Incorrect results! D",
                                        1e-4,
                                        1e-5);
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
