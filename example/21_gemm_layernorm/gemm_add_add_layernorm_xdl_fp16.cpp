// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_layernorm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

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
using AddReluAdd  = ck::tensor_operation::element_wise::AddReluAdd;

// DataType
using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F16;
using D1DataType       = F16;
using DsDataType       = ck::Tuple<D0DataType, D1DataType>;
using GammaDataType    = F16;
using BetaDataType     = F16;
using HDataType        = F16;

// Layout
using ALayout  = Row;
using BLayout  = Col;
using D0Layout = Row;
using D1Layout = Row;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using HLayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddReluAdd;
using HElementOp   = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleDLayernorm_Xdl_CShuffle
//######| ALayout| BLayout| DsLayout| HLayout|     AData|     BData|     AccData|         CShuffle|     DsData|     GammaData|     BetaData|     HData|           A|           B|          CDE|            H|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|           PostShuffle|     PostShuffle|
//######|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|          Type|         Type|      Type| Elementwise| Elementwise|  Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|        ClusterLengths| ScalarPerVector|
//######|        |        |         |        |          |          |            |                 |           |              |             |          |   Operation|   Operation|    Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|                  _M_N|            _M_N|
//######|        |        |         |        |          |          |            |                 |           |              |             |          |            |            |             |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                      |                | LayernormThreadClusterSize_M_N, LayernormThreadSliceSize_M_N
        < ALayout, BLayout, DsLayout, HLayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, GammaDataType, BetaDataType, HDataType,  AElementOp,  BElementOp, CDEElementOp,   HElementOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,              S<64, 4>,               4, S<8, 32>, S<1, 8>, 1, 8, 8, 8, 8, 1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        AccDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        PassThrough>;

auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
    return HostTensorDescriptor(std::vector<std::size_t>({len}),
                                std::vector<std::size_t>({stride}));
};

auto f_host_tensor_descriptor2d =
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

int main()
{
    bool do_verification = true;

    // GEMM shape
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA  = 1024;
    ck::index_t StrideB  = 1024;
    ck::index_t StrideD0 = 0;
    ck::index_t StrideD1 = 1024;
    ck::index_t StrideH  = 1024;

    float epsilon = 1e-5;

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<D0DataType> d0_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor2d(M, N, StrideD1, D1Layout{}));
    Tensor<GammaDataType> gamma_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<BetaDataType> beta_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<HDataType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideH, HLayout{}));
    Tensor<HDataType> h_m_n(f_host_tensor_descriptor2d(M, N, StrideH, HLayout{}));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-1, 1});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-1, 1});
    d0_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-1, 1});
    d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{-1, 1});
    gamma_n.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{-1, 1});
    beta_n.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{-1, 1});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpaceSize());
    DeviceMem gamma_device_buf(sizeof(GammaDataType) * gamma_n.mDesc.GetElementSpaceSize());
    DeviceMem beta_device_buf(sizeof(BetaDataType) * beta_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(HDataType) * e_m_n.mDesc.GetElementSpaceSize());
    DeviceMem h_device_buf(sizeof(HDataType) * h_m_n.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    d0_device_buf.ToDevice(d0_n.mData.data());
    d1_device_buf.ToDevice(d1_m_n.mData.data());
    gamma_device_buf.ToDevice(gamma_n.mData.data());
    beta_device_buf.ToDevice(beta_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto h_element_op   = HElementOp{};

    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                               b_device_buf.GetDeviceBuffer(),
                               {d0_device_buf.GetDeviceBuffer(), d1_device_buf.GetDeviceBuffer()},
                               gamma_device_buf.GetDeviceBuffer(),
                               beta_device_buf.GetDeviceBuffer(),
                               e_device_buf.GetDeviceBuffer(),
                               h_device_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               {StrideD0, StrideD1},
                               StrideH,
                               epsilon,
                               a_element_op,
                               b_element_op,
                               cde_element_op,
                               h_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    size_t workspace_sz = device_op.GetWorkSpaceSize(&argument);
    DeviceMem workspace_dev(workspace_sz);
    device_op.SetWorkSpacePointer(&argument, workspace_dev.GetDeviceBuffer());

    invoker.Run(argument, StreamConfig{nullptr, false});

    if(do_verification)
    {
        Tensor<AccDataType> c_m_n_host(HostTensorDescriptor{M, N});
        Tensor<HDataType> e_m_n_host(HostTensorDescriptor{M, N});

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host(m, n), c_m_n_host(m, n), d0_n(n), d1_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n.mData.data());
        return ck::utils::check_err(e_m_n, e_m_n_host) ? 0 : 1;
    }
}
