// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_multiple_r_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;
using F64 = double;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

// DataType
using ADataType         = F16;
using BDataType         = F16;
using GemmAccDataType   = F32;
using CShuffleDataType  = F32;
using D0DataType        = F16;
using D1DataType        = F16;
using DsDataType        = ck::Tuple<D0DataType, D1DataType>;
using EDataType         = F16;
using ReduceAccDataType = F32;
using R0DataType        = F32;
using R1DataType        = F32;
using RsDataType        = ck::Tuple<R0DataType, R1DataType>;

// Layout
using ALayout  = Row;
using BLayout  = Col;
using D1Layout = Row;
using ELayout  = D1Layout;

// Elementwise op
using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AddAdd       = ck::tensor_operation::element_wise::AddAdd;
using Square       = ck::tensor_operation::element_wise::UnarySquare;
using Div          = ck::tensor_operation::element_wise::UnaryDivide;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddAdd;
using QsElementOp  = ck::Tuple<PassThrough, Square>;
using RsElementOp  = ck::Tuple<Div, Div>;

// ReduceOp
using R0ThreadReduceOp = ck::reduce::Add;
using R1ThreadReduceOp = ck::reduce::Add;
using RsThreadReduceOp = ck::Tuple<R0ThreadReduceOp, R1ThreadReduceOp>;

static constexpr auto R0GlobalReduceOp = ck::InMemoryDataOperationEnum::AtomicAdd;
static constexpr auto R1GlobalReduceOp = ck::InMemoryDataOperationEnum::AtomicAdd;
using RsGlobalReduceOp = ck::InMemoryDataOperationEnumSequence<R0GlobalReduceOp, R1GlobalReduceOp>;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleDMultipleR_Xdl_CShuffle
//######| ALayout| BLayout| ELayout|     AData|     BData|     GemmAccData|         CShuffle|     DsData|     EData|     ReduceAccData|     RsData|           A|           B|          CDE|          Qs|          Rs|           Thread|           Global|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|    CDRThreadTransfer|                  CDE|    RThreadTransfer|
//######|        |        |        |      Type|      Type|            Type|         DataType|       Type|      Type|              Type|       Type| Elementwise| Elementwise|  Elementwise| Elementwise| Elementwise|           Reduce|           Reduce| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|       ClusterLengths| ReduceThreadTransfer| DstScalarPerVector|
//######|        |        |        |          |          |                |                 |           |          |                  |           |   Operation|   Operation|    Operation|   Operation|   Operation|        Operation|        Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _MPerBlock_NPerBlock|      ScalarPerVector|         _MPerBlock|
//######|        |        |        |          |          |                |                 |           |          |                  |           |            |            |             |            |            |                 |                 |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                     |           _NPerBlock|                   |
        < ALayout, BLayout, ELayout, ADataType, BDataType, GemmAccDataType, CShuffleDataType, DsDataType, EDataType, ReduceAccDataType, RsDataType,  AElementOp,  BElementOp, CDEElementOp, QsElementOp, RsElementOp, RsThreadReduceOp, RsGlobalReduceOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<64, 4>,                    4,                  1>;
// clang-format on

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

int main(int argc, char* argv[])
{
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA  = 1024;
    ck::index_t StrideB  = 1024;
    ck::index_t StrideD0 = 0;
    ck::index_t StrideD1 = 1024;
    ck::index_t StrideE  = 1024;

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<D0DataType> d0_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor2d(M, N, StrideD1, ELayout{}));
    Tensor<EDataType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, D1Layout{}));
    Tensor<R0DataType> r0_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<R1DataType> r1_m(f_host_tensor_descriptor1d(M, 1));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-1, 1});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-1, 1});
    d0_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-1, 1});
    d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{-5, 5});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_n.mDesc.GetElementSpace());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpace());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n.mDesc.GetElementSpace());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_m.mDesc.GetElementSpace());
    DeviceMem r1_device_buf(sizeof(R1DataType) * r1_m.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    d0_device_buf.ToDevice(d0_n.mData.data());
    d1_device_buf.ToDevice(d1_m_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto qs_element_op  = QsElementOp{};
    auto rs_element_op  = RsElementOp{N, N};

    // Prepare GEMM, mean, mean_square
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(
        a_device_buf.GetDeviceBuffer(),
        b_device_buf.GetDeviceBuffer(),
        std::array<const void*, 2>{d0_device_buf.GetDeviceBuffer(),
                                   d1_device_buf.GetDeviceBuffer()},
        e_device_buf.GetDeviceBuffer(),
        std::array<void*, 2>{r0_device_buf.GetDeviceBuffer(), r1_device_buf.GetDeviceBuffer()},
        M,
        N,
        K,
        StrideA,
        StrideB,
        std::array<ck::index_t, 2>{StrideD0, StrideD1},
        StrideE,
        a_element_op,
        b_element_op,
        cde_element_op,
        qs_element_op,
        rs_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, false});

    return 0;
}
