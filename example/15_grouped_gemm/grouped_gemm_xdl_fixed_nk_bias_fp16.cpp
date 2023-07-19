// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddBias     = ck::tensor_operation::element_wise::AddBias;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType>;
using EDataType        = F16;

using ALayout  = Row;
using BLayout  = Col;
using D0Layout = Row;
using DsLayout = ck::Tuple<D0Layout>;
using ELayout  = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;

using CDEElementOp = AddBias;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNPadding;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGroupedGemm_Xdl_Fixed_NK
    // clang-format off
//######| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
//######|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
//######|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
//######|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>;
// clang-format on

struct ProblemSize final
{
    std::vector<ck::index_t> Ms;
    std::vector<ck::index_t> Ns;
    std::vector<ck::index_t> Ks;

    std::vector<ck::index_t> stride_As;
    std::vector<ck::index_t> stride_Bs;
    std::vector<ck::index_t> stride_Cs;

    ck::index_t group_count;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

bool run_grouped_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    auto group_count = problem_size.group_count;

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    std::vector<void*> p_Cs;

    gemm_descs.reserve(group_count);

    int sum_of_m = 0;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<D0DataType>> d0_tensors;
    std::vector<Tensor<EDataType>> c_host_tensors;
    std::vector<Tensor<EDataType>> c_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    d0_tensors.reserve(group_count);
    c_host_tensors.reserve(group_count);
    c_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, d0_tensors_device,
        c_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    d0_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(int i = 0; i < group_count; i++)
    {
        sum_of_m += problem_size.Ms[i];
        a_tensors.push_back(Tensor<ADataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ks[i], problem_size.stride_As[i], ALayout{})));
        b_tensors.push_back(Tensor<BDataType>(f_host_tensor_descriptor(
            problem_size.Ks[i], problem_size.Ns[i], problem_size.stride_Bs[i], BLayout{})));
        d0_tensors.push_back(Tensor<D0DataType>(
            f_host_tensor_descriptor(problem_size.Ms[i], problem_size.Ns[i], 0, ELayout{})));
        c_host_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ns[i], problem_size.stride_Cs[i], ELayout{})));
        c_device_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ns[i], problem_size.stride_Cs[i], ELayout{})));
        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_k_n: " << b_tensors[i].mDesc << " d_m_n: " << d0_tensors[i].mDesc
                  << " c_m_n: " << c_device_tensors[i].mDesc << std::endl;

        flop += std::size_t(2) * problem_size.Ms[i] * problem_size.Ks[i] * problem_size.Ns[i];
        num_btype += sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize() +
                     sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(D0DataType) * d0_tensors[i].mDesc.GetElementSize() +
                     sizeof(EDataType) * c_device_tensors[i].mDesc.GetElementSize();

        switch(config.init_method)
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

        d0_tensors[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    using GroupedGemmKernelArgument = ck::tensor_operation::device::GroupedGemmKernelArgument<1>;

    std::vector<GroupedGemmKernelArgument> grouped_gemm_kernel_args_;
    grouped_gemm_kernel_args_.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        a_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ADataType) * sum_of_m * problem_size.Ks[i]));

        b_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(BDataType) * problem_size.Ns[i] * problem_size.Ks[i]));

        d0_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(D0DataType) * problem_size.Ns[i]));

        c_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(EDataType) * sum_of_m * problem_size.Ns[i]));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data(),
                                      a_tensors[i].mDesc.GetElementSpaceSize() * sizeof(ADataType));
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data(),
                                      b_tensors[i].mDesc.GetElementSpaceSize() * sizeof(BDataType));
        d0_tensors_device[i]->ToDevice(d0_tensors[i].mData.data());
        c_tensors_device[i]->SetZero();

        p_Cs.push_back(c_tensors_device[i]->GetDeviceBuffer());

        gemm_descs.push_back(
            {1, problem_size.Ns[i], problem_size.Ks[i], 1, problem_size.stride_Bs[i], 1, {0}});

        grouped_gemm_kernel_args_.push_back(
            {a_tensors_device[i]->GetDeviceBuffer(),
             b_tensors_device[i]->GetDeviceBuffer(),
             std::array<const void*, 1>{d0_tensors_device[i]->GetDeviceBuffer()},
             c_tensors_device[i]->GetDeviceBuffer(),
             problem_size.Ms[i],
             problem_size.Ns[i],
             problem_size.Ks[i],
             problem_size.stride_As[i],
             problem_size.stride_Bs[i],
             std::array<ck::index_t, 1>{0},
             problem_size.stride_Cs[i]});
    }

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    std::vector<const void*> p_As                = {};
    std::vector<const void*> p_Bs                = {};
    std::vector<std::array<const void*, 1>> p_Ds = {};

    // do GEMM
    auto argument = gemm.MakeArgument(
        p_As, p_Bs, p_Ds, p_Cs, gemm_descs, a_element_op, b_element_op, cde_element_op);

    DeviceMem gemm_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    hip_check_error(hipMemcpy(gemm_desc_workspace.GetDeviceBuffer(),
                              grouped_gemm_kernel_args_.data(),
                              gemm.GetWorkSpaceSize(&argument),
                              hipMemcpyHostToDevice));

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    gemm.SetDeviceKernelArgs(argument, gemm_desc_workspace.GetDeviceBuffer());

    invoker.Run(argument, StreamConfig{nullptr, false});

    bool pass = true;
    if(config.do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                EDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                PassThrough>;

        for(std::size_t i = 0; i < gemm_descs.size(); i++)
        {
            c_tensors_device[i]->FromDevice(c_device_tensors[i].mData.data(),
                                            c_device_tensors[i].mDesc.GetElementSize() *
                                                sizeof(EDataType));

            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                      b_tensors[i],
                                                      c_host_tensors[i],
                                                      a_element_op,
                                                      b_element_op,
                                                      PassThrough{});

            ref_invoker.Run(ref_argument);

            for(int m = 0; m < problem_size.Ms[i]; ++m)
            {
                for(int n = 0; n < problem_size.Ns[i]; ++n)
                {
                    cde_element_op(
                        c_host_tensors[i](m, n), c_host_tensors[i](m, n), d0_tensors[i](m, n));
                }
            }

            pass &= ck::utils::check_err(c_device_tensors[i], c_host_tensors[i]);
        }
    }

    if(config.time_kernel)
    {
        float ave_time   = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
        float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }

    return pass;
}

// int main(int argc, char* argv[]) { return !run_grouped_gemm_example(argc, argv); }
int main(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    problem_size.group_count = 16;

    problem_size.Ms = {167, 0, 177, 181, 153, 0, 156, 173, 645, 150, 204, 184, 168, 156, 168, 148};

    for(int i = 0; i < problem_size.group_count; i++)
    {
        problem_size.Ns.push_back(768);
        problem_size.Ks.push_back(4608);

        problem_size.stride_As.push_back(problem_size.Ks[i]);
        problem_size.stride_Bs.push_back(problem_size.Ks[i]);
        problem_size.stride_Cs.push_back(problem_size.Ns[i]);
    }

    if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        exit(0);
    }

    return !run_grouped_gemm(problem_size, config);
}
