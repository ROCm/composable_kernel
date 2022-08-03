// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_contraction_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using DsDataType       = ck::Tuple<>;
using EDataType        = F16;

static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr auto ABSpec = ck::tensor_operation::device::TensorSpecialization::Packed;
static constexpr auto DESpec = ck::tensor_operation::device::TensorSpecialization::Default;

// clang-format off
using DeviceOpInstanceKKNN = ck::tensor_operation::device::
        //############################################| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|     DsData| EData|            A|           B|          CDE|           Gemm|              A|              B|             DE| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################################|        |        |        |  Type|  Type|    Type| DataType|       Type|  Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Spacialization| Spacialization| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################################|        |        |        |      |      |        |         |           |      |    Operation|   Operation|    Operation|               |               |               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################################|        |        |        |      |      |        |         |           |      |             |            |             |               |               |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedContractionMultipleD_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F16,   F16,     F32,      F16, DsDataType,   F16,   AElementOp,  BElementOp, CDEElementOp,       GemmSpec,         ABSpec,         ABSpec,         DESpec,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,              S<1, 32, 1, 4>,               1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        EDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CDEElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        exit(0);
    }

    int group_count = rand() % 16 + 1;

    // GEMM shape
    std::vector<ck::tensor_operation::device::ContractionDesc<0>> contraction_descs;
    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_c;

    contraction_descs.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        int M = 256 + 256 * i;
        int N = 128 + 128 * i;
        int K = 64 + 64 * i;

        // A[M, K]
        std::vector<ck::index_t> a_ms_ks_lengths{M, K};
        std::vector<ck::index_t> a_ms_ks_strides{K, 1};
        // B[N, K]
        std::vector<ck::index_t> b_ns_ks_lengths{N, K};
        std::vector<ck::index_t> b_ns_ks_strides{K, 1};
        // E[M, M]
        std::vector<ck::index_t> e_ms_ns_lengths{M, N};
        std::vector<ck::index_t> e_ms_ns_strides{N, 1};

        contraction_descs.push_back(ck::tensor_operation::device::ContractionDesc<0>{
            a_ms_ks_lengths,
            a_ms_ks_strides,
            b_ns_ks_lengths,
            b_ns_ks_strides,
            std::array<std::vector<ck::index_t>, 0>{},
            std::array<std::vector<ck::index_t>, 0>{},
            e_ms_ns_lengths,
            e_ms_ns_strides});
    }

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<EDataType>> e_host_tensors;
    std::vector<Tensor<EDataType>> e_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);

    e_host_tensors.reserve(group_count);
    e_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, c_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(std::size_t i = 0; i < contraction_descs.size(); i++)
    {
        const auto a_ms_ks_lengths = contraction_descs[i].a_ms_ns_lengths;
        const auto a_ms_ks_strides = contraction_descs[i].a_ms_ks_strides;

        const auto b_ns_ks_lengths = contraction_descs[i].b_ns_ks_lengths;
        const auto b_ns_ks_strides = contraction_descs[i].b_ns_ks_strides;

        const auto e_ms_ns_lengths = contraction_descs[i].e_ms_ns_lengths;
        const auto e_ms_ns_strides = contraction_descs[i].e_ms_ns_strides;

        Tensor<ADataType> a_ms_ks(
            std::vector<std::size_t>(a_ms_ks_lengths.begin(), a_ms_ks_lengths.end()),
            std::vector<std::size_t>(a_ms_ks_strides.begin(), a_ms_ks_strides.end()));
        Tensor<BDataType> b_ns_ks(
            std::vector<std::size_t>(b_ns_ks_lengths.begin(), b_ns_ks_lengths.end()),
            std::vector<std::size_t>(b_ns_ks_strides.begin(), b_ns_ks_strides.end()));

        Tensor<EDataType> e_ms_ns_host_result(
            std::vector<std::size_t>(e_ms_ns_lengths.begin(), e_ms_ns_lengths.end()),
            std::vector<std::size_t>(e_ms_ns_strides.begin(), e_ms_ns_strides.end()));
        Tensor<EDataType> e_ms_ns_device_result(
            std::vector<std::size_t>(e_ms_ns_lengths.begin(), e_ms_ns_lengths.end()),
            std::vector<std::size_t>(e_ms_ns_strides.begin(), e_ms_ns_strides.end()));

        ck::index_t M_ = std::accumulate(e_ms_ns_lengths.begin(),
                                         e_ms_ns_lengths.begin() + NumDimM,
                                         ck::index_t{1},
                                         std::multiplies<ck::index_t>{});

        ck::index_t N_ = std::accumulate(e_ms_ns_lengths.begin() + NumDimM,
                                         e_ms_ns_lengths.begin() + NumDimM + NumDimN,
                                         ck::index_t{1},
                                         std::multiplies<ck::index_t>{});

        ck::index_t K_ = std::accumulate(a_ms_ks_lengths.begin() + NumDimM,
                                         a_ms_ks_lengths.begin() + NumDimM + NumDimK,
                                         ck::index_t{1},
                                         std::multiplies<ck::index_t>{});

        a_tensors.push_back(a_ms_ks);
        b_tensors.push_back(b_ns_ks);

        e_host_tensors.push_back(e_ms_ns_host_result);
        e_device_tensors.push_back(e_ms_ns_device_result);

        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_n_k: " << b_tensors[i].mDesc << " c_m_n: " << e_device_tensors[i].mDesc
                  << std::endl;

        flop += std::size_t(2) * M_ * K_ * N_;
        num_btype += sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize() +
                     sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(EDataType) * e_device_tensors[i].mDesc.GetElementSize();

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

    for(std::size_t i = 0; i < contraction_descs.size(); i++)
    {
        a_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(ADataType) * a_tensors[i].mDesc.GetElementSpaceSize()));
        b_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(BDataType) * b_tensors[i].mDesc.GetElementSpaceSize()));
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(EDataType) * e_device_tensors[i].mDesc.GetElementSpaceSize()));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CDEElementOp{};

    auto gemm    = DeviceOpInstanceKKNN{};
    auto invoker = gemm.MakeInvoker();

    std::vector<std::array<const void*, 0>> p_Ds = {};

    // do GEMM
    auto argument = gemm.MakeArgument(
        p_a, p_b, p_Ds, p_c, contraction_descs, a_element_op, b_element_op, c_element_op);

    DeviceMem contraction_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, contraction_desc_workspace.GetDeviceBuffer());

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
        for(std::size_t i = 0; i < contraction_descs.size(); i++)
        {
            c_tensors_device[i]->FromDevice(e_device_tensors[i].mData.data());
            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                      b_tensors[i],
                                                      e_host_tensors[i],
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op);

            ref_invoker.Run(ref_argument);
            pass &= ck::utils::check_err(e_device_tensors[i].mData, e_host_tensors[i].mData);
        }
    }

    return pass ? 0 : 1;
}
