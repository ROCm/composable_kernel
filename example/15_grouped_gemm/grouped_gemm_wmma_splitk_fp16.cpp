// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <tuple>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/utility/ignore.hpp"
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

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = F16;

using ALayout  = Row;
using BLayout  = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGroupedGemmWmmaSplitKCShuffle
    // clang-format off
//######| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
//######|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|         _MBlock_MWaveMPerXdl| ScalarPerVector|
//######|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
//######|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   128,   128,    64,   8,   8,   16,   16,       2,       4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8>;

// clang-format on

#define EXAMPLE_USE_SPLITK
#include "run_grouped_gemm_example.inc"

template <typename T>
void print_tensor(const Tensor<T>& tensor)
{
    for(size_t y = 0; y < tensor.GetLengths()[1]; ++y)
    {
        for(size_t x = 0; x < tensor.GetLengths()[0]; ++x)
        {
            std::cout << ck::type_convert<float>(tensor(y, x)) << " ";
        }

        std::cout << std::endl;
    }
}

bool run_grouped_gemm_test(std::vector<ck::tensor_operation::device::GemmDesc>& gemm_descs)
{
    int group_count = gemm_descs.size();

    // GEMM shape

    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_c;

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
    std::vector<Tensor<EDataType>> c_host_tensors;
#ifdef BUILD_INT4_EXAMPLE
    std::vector<Tensor<KernelEDataType>> c_device_tensors;
#else
    std::vector<Tensor<EDataType>> c_device_tensors;
#endif

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    c_host_tensors.reserve(group_count);
    c_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, c_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    for(std::size_t i = 0; i < gemm_descs.size(); i++)
    {
        a_tensors.push_back(Tensor<ADataType>(f_host_tensor_descriptor(
            gemm_descs[i].M_, gemm_descs[i].K_, gemm_descs[i].stride_A_, ALayout{})));
        b_tensors.push_back(Tensor<BDataType>(f_host_tensor_descriptor(
            gemm_descs[i].K_, gemm_descs[i].N_, gemm_descs[i].stride_B_, BLayout{})));
        c_host_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            gemm_descs[i].M_, gemm_descs[i].N_, gemm_descs[i].stride_C_, ELayout{})));
#ifdef BUILD_INT4_EXAMPLE
        c_device_tensors.push_back(Tensor<KernelEDataType>(f_host_tensor_descriptor(
            gemm_descs[i].M_, gemm_descs[i].N_, gemm_descs[i].stride_C_, ELayout{})));
#else
        c_device_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            gemm_descs[i].M_, gemm_descs[i].N_, gemm_descs[i].stride_C_, ELayout{})));
#endif
        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_k_n: " << b_tensors[i].mDesc << " c_m_n: " << c_device_tensors[i].mDesc
                  << std::endl;

        // a_tensors[i].GenerateTensorValue(GeneratorTensor_Diagonal<ADataType>{});
        // b_tensors[i].GenerateTensorValue(GeneratorTensor_Diagonal<BDataType>{});
        a_tensors[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_tensors[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});

        //     a_tensors[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        //     b_tensors[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    for(std::size_t i = 0; i < gemm_descs.size(); i++)
    {
        a_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(ADataType) * a_tensors[i].mDesc.GetElementSpaceSize()));
        b_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(BDataType) * b_tensors[i].mDesc.GetElementSpaceSize()));
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(EDataType) * c_device_tensors[i].mDesc.GetElementSpaceSize()));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());
        c_tensors_device[i]->SetZero();

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CDEElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    std::vector<std::array<const void*, 0>> p_Ds = {};

    // do GEMM
    // auto argument = gemm.MakeArgument(
    //     p_a, p_b, p_Ds, p_c, gemm_descs, a_element_op, b_element_op, c_element_op);
    auto argument = DeviceGemmInstance::Argument(p_a, p_b, p_c, gemm_descs, 4);

    std::size_t workspace_size = gemm.GetWorkSpaceSize(&argument);
    std::size_t kargs_size     = gemm.GetDeviceKernelArgSize(&argument);

    DeviceMem gemm_workspace, gemm_kargs;

    // The following is necessary since TwoStage kernel is using additional memory both
    // for Workspace and kernel arguments.
    if(kargs_size > 0)
    {
        gemm_kargs.Realloc(kargs_size);
        gemm.SetDeviceKernelArgs(&argument, gemm_kargs.GetDeviceBuffer());
    }
    if(workspace_size > 0 && workspace_size != kargs_size)
    {
        gemm_workspace.Realloc(workspace_size);
        gemm.SetWorkSpacePointer(&argument, gemm_workspace.GetDeviceBuffer());
    }

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    invoker.Run(argument, StreamConfig{nullptr, false});

    bool pass                   = true;
    using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                            BDataType,
                                                                            EDataType,
                                                                            AccDataType,
                                                                            AElementOp,
                                                                            BElementOp,
                                                                            CDEElementOp>;

    for(std::size_t i = 0; i < gemm_descs.size(); i++)
    {
        c_tensors_device[i]->FromDevice(c_device_tensors[i].mData.data());
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                  b_tensors[i],
                                                  c_host_tensors[i],
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op);

        ref_invoker.Run(ref_argument);
        pass &= ck::utils::check_err(c_device_tensors[i], c_host_tensors[i]);

        // std::cout << "A Tensor:" << std::endl;
        // print_tensor(a_tensors[i]);
        // std::cout << std::endl;

        // std::cout << "B Tensor:" << std::endl;
        // print_tensor(b_tensors[i]);
        // std::cout << std::endl;

        // std::cout << "C Tensor:" << std::endl;
        // print_tensor(c_device_tensors[i]);
        // std::cout << std::endl;

        // std::cout << "CPU reference tensor:" << std::endl;
        // print_tensor(c_host_tensors[i]);
        // std::cout << std::endl;
    }

    return pass;
}

bool grouped_gemm_test(int argc, char* argv[])
{
    // Lambda to get stride based on layout
    auto get_stride = [](auto layout, auto row_dim, auto col_dim) {
        if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
        {
            return col_dim;
        }
        else
        {
            return row_dim;
        }
    };

    using GemmDesc = ck::tensor_operation::device::GemmDesc;
    std::vector<GemmDesc> gemm_descs;

    for(size_t i = 0; i < 16; ++i)
    {
        ck::index_t M = 256 + 256 * i;
        ck::index_t N = 128 + 128 * i;
        ck::index_t K = 128 + 64 * i;

        gemm_descs.push_back(GemmDesc{M,
                                      N,
                                      K,
                                      get_stride(ALayout{}, M, K),
                                      get_stride(BLayout{}, K, N),
                                      get_stride(ELayout{}, M, N),
                                      std::vector<ck::index_t>()});
    }

    ck::ignore = argc;
    ck::ignore = argv;

    return run_grouped_gemm_test(gemm_descs);
}

int main(int argc, char* argv[]) { return !run_grouped_gemm_example(argc, argv); }
