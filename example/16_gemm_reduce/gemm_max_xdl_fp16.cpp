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
using DsDataType        = ck::Tuple<>;
using EDataType         = F16;
using ReduceAccDataType = F32;
using R0DataType        = F32;
using RsDataType        = ck::Tuple<R0DataType>;

// Layout
using ALayout = Row;
using BLayout = Col;
using ELayout = Row;

// Elementwise op
using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;
using QsElementOp  = ck::Tuple<PassThrough>;
using RsElementOp  = ck::Tuple<PassThrough>;

// ReduceOp
using RsThreadReduceOp = ck::Tuple<ck::reduce::Max>;

using RsGlobalReduceOp =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicMax>;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleDMultipleR_Xdl_CShuffle
//######| ALayout| BLayout| ELayout|     AData|     BData|     GemmAccData|         CShuffle|     DsData|     EData|     ReduceAccData|     RsData|           A|           B|          CDE|          Qs|          Rs|           Thread|           Global|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|    CDRThreadTransfer|                  CDE|    RThreadTransfer|
//######|        |        |        |      Type|      Type|            Type|         DataType|       Type|      Type|              Type|       Type| Elementwise| Elementwise|  Elementwise| Elementwise| Elementwise|           Reduce|           Reduce| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|       ClusterLengths| ReduceThreadTransfer| DstScalarPerVector|
//######|        |        |        |          |          |                |                 |           |          |                  |           |   Operation|   Operation|    Operation|   Operation|   Operation|        Operation|        Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _MPerBlock_NPerBlock|      ScalarPerVector|         _MPerBlock|
//######|        |        |        |          |          |                |                 |           |          |                  |           |            |            |             |            |            |                 |                 |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                     |           _NPerBlock|                   |
        < ALayout, BLayout, ELayout, ADataType, BDataType, GemmAccDataType, CShuffleDataType, DsDataType, EDataType, ReduceAccDataType, RsDataType,  AElementOp,  BElementOp, CDEElementOp, QsElementOp, RsElementOp, RsThreadReduceOp, RsGlobalReduceOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<64, 4>,                    4,                  1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        EDataType,
                                                                        GemmAccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CDEElementOp>;

template <typename ADataType, typename BDataType, typename EDataType, typename R0DataType>
void DumpPerf(float ave_time, int M, int N, int K)
{
    std::size_t flop          = std::size_t(2) * M * N * K;
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(EDataType) * M * N + sizeof(R0DataType) * M;

    float tflops          = static_cast<float>(flop) / 1.E9 / ave_time;
    float gemm_gb_per_sec = gemm_num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gemm_gb_per_sec
              << " GB/s, " << std::endl;
}

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
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA = 1024;
    ck::index_t StrideB = 1024;
    ck::index_t StrideE = 1024;

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<EDataType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));
    Tensor<R0DataType> r0_m(f_host_tensor_descriptor1d(M, 1));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-1, 1});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-1, 1});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n.mDesc.GetElementSpace());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_m.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto qs_element_op  = QsElementOp{};
    auto rs_element_op  = RsElementOp{};

    // Prepare GEMM, max
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                           b_device_buf.GetDeviceBuffer(),
                                           {},
                                           e_device_buf.GetDeviceBuffer(),
                                           {r0_device_buf.GetDeviceBuffer()},
                                           M,
                                           N,
                                           K,
                                           StrideA,
                                           StrideB,
                                           {},
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

    // [CAUSION]: launch_and_time_kernel will not initialize D.
    // If we evaluate kernel multiple time but without initialize D. Verification will fail
    r0_device_buf.SetValue(ck::NumericLimits<R0DataType>::Lowest());

    invoker.Run(argument, StreamConfig{nullptr, false});

    bool do_verification = true;
    bool pass            = true;

    if(do_verification)
    {
        auto I0 = ck::Number<0>{};

        Tensor<EDataType> e_m_n_host(e_m_n.mDesc);
        Tensor<R0DataType> r0_m_host(r0_m.mDesc);

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, e_m_n_host, a_element_op, b_element_op, cde_element_op);

        ref_invoker.Run(ref_argument);

        auto reduce0_op = RsThreadReduceOp{}[I0];

        for(int m = 0; m < M; ++m)
        {
            auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                auto e_val = ck::type_convert<ReduceAccDataType>(e_m_n_host(m, n));
                reduce0_op(reduce0_acc, e_val);
            };

            r0_m_host(m) = ck::type_convert<R0DataType>(reduce0_acc);
        }

        e_device_buf.FromDevice(e_m_n.mData.data());
        r0_device_buf.FromDevice(r0_m.mData.data());

        pass = ck::utils::check_err(
            e_m_n.mData, e_m_n_host.mData, "Error: Incorrect results c", 1e-2, 1e-2);
        pass &= ck::utils::check_err(
            r0_m.mData, r0_m_host.mData, "Error: Incorrect results d0", 1e-2, 1e-2);
    }

    bool time_kernel = true;
    if(time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});
        DumpPerf<ADataType, BDataType, EDataType, R0DataType>(ave_time, M, N, K);
    }

    return pass ? 0 : 1;
}
