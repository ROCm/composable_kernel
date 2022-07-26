// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_bias_add_reduce_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
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

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType            = F16;
using BDataType            = F16;
using CDataType            = F16;
using BiasDataType         = F32;
using D0DataType           = F16;
using GemmAccDataType      = F32;
using ReduceAccDataType    = F32;
using ReduceDataType       = F32;
using ReducePtrsGlobal     = ck::Tuple<ReduceDataType*, ReduceDataType*>;
using GammaDataType        = F16;
using BetaDataType         = F16;
using LayerNormOutDataType = F16;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AElementOp  = PassThrough;
using BElementOp  = PassThrough;
using CElementOp  = ck::tensor_operation::element_wise::Relu;
using D0ElementOp = PassThrough;
using ReduceSumOp = ck::reduce::Add;
using ReduceOps   = ck::Tuple<ReduceSumOp, ReduceSumOp>;

using UnaryIdenticElementOp = ck::tensor_operation::element_wise::PassThrough;
using UnaryDivElementOp     = ck::tensor_operation::element_wise::UnaryDivide;
using UnarySquareElementOp  = ck::tensor_operation::element_wise::UnarySquare;
using ReduceInElementOps    = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
using ReduceOutElementOps   = ck::Tuple<UnaryDivElementOp, UnaryDivElementOp>;

using ReduceGlobalMemOps =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicAdd,
                                          ck::InMemoryDataOperationEnum::AtomicAdd>;

static constexpr auto GemmSpecialization =
    ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmBiasAddReduceInstance = ck::tensor_operation::device::DeviceGemmBiasAddReduce_Xdl_CShuffle
//######| ALayout| BLayout| CLayout|AData| BData| CData|C0Data|C1Data|  GemmAcc| CShuffle| ReduceAcc|       ReduceData|           A|           B|           C|          C1|    Reduce|     ReduceInEleOp|      ReduceAccEleOp|              Reduce|               GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|              CReduce| CReduceThreadLds2VGprCopy| CReduceThreadVgpr2GlobalCopy|
//######|        |        |        | Type|  Type|  Type|  Type|  Type| DataType| DataType|  DataType|       Type Tuple| Elementwise| Elementwise| Elementwise| Elementwise| Operation|                  |                    |          MemoryData|     Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN| MXdlPerWave| NXdlPerWave|            _MBlock_MPerBlock| ScalarPerVector| ThreadClusterLengths|     SrcDstScalarPerVector|        SrcDstScalarPerVector|
//######|        |        |        |     |      |      |      |      |         |         |          |                 |   Operation|   Operation|   Operation|   Operation|          |                  |                    |           Operation|                   |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|            _NBlock_NPerBlock|      _NPerBlock| _MPerBlock_NPerBlock|                _NPerBlock|                   _MPerBlock|
//######|        |        |        |     |      |      |      |      |         |         |          |                 |            |            |            |            |          |                  |                    |                    |                   |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                     |                          |                             |
        <     Row,     Col,     Row,  F16,   F16,   F16,   F32,   F16,      F32,      F32,       F32, ReducePtrsGlobal,  AElementOp,  BElementOp,  CElementOp, D0ElementOp, ReduceOps,ReduceInElementOps, ReduceOutElementOps,  ReduceGlobalMemOps, GemmSpecialization,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8,             S<64, 4>,                         4,                            1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        GemmAccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        PassThrough>;

using NormalizeFunctor = ck::tensor_operation::element_wise::Normalize;

// A:x, B:E[x], C:E[x^2], D:Gamma, E:Beta , F:y
using DeviceNormalizeInstance = ck::tensor_operation::device::DeviceElementwise<
    ck::Tuple<CDataType,
              ReduceDataType,
              ReduceDataType,
              GammaDataType,
              BetaDataType>,         // x(gemm_out), mean, meansquare, gamma, beta
    ck::Tuple<LayerNormOutDataType>, // y
    NormalizeFunctor,
    2,
    8,                           // MPerthread
    ck::Sequence<8, 1, 1, 8, 8>, // scalarPerVector: x(gemm_out), mean, meansquare, gamma, beta
    ck::Sequence<8>>;            // scalarPerVector: y(layerNorm_out)

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

template <typename CDataType,
          typename ReduceDataType,
          typename AccDataType,
          typename BiasDataType,
          typename D0DataType,
          typename A_functor,
          typename B_functor,
          typename C_functor,
          typename C1_functor>
void host_gemm_layernorm(Tensor<LayerNormOutDataType>& out_m_n,
                         const Tensor<ADataType>& a_m_k,
                         const Tensor<ADataType>& b_k_n,
                         const Tensor<BiasDataType>& bias_n,
                         const Tensor<D0DataType>& c1_m_n,
                         const Tensor<GammaDataType>& gamma_n,
                         const Tensor<GammaDataType>& beta_n,
                         A_functor a_element_op,
                         B_functor b_element_op,
                         C_functor c_element_op,
                         C1_functor c1_element_op,
                         int M,
                         int N)
{

    int StrideC = N;
    Tensor<CDataType> c_m_n(f_host_tensor_descriptor2d(M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> mean_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<ReduceDataType> meanSquare_m(f_host_tensor_descriptor1d(M, 1));
    auto averageOpInst = UnaryDivElementOp{N};

    auto ref_gemm    = ReferenceGemmInstance{};
    auto ref_invoker = ref_gemm.MakeInvoker();

    auto ref_argument =
        ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, a_element_op, b_element_op, PassThrough{});

    ref_invoker.Run(ref_argument);

    // c = activation(c + bias) + c1_functor(c1)
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
        {
            AccDataType acc = ck::type_convert<AccDataType>(c_m_n(m, n)) +
                              ck::type_convert<AccDataType>(bias_n(n));

            AccDataType c1 = ck::type_convert<AccDataType>(c1_m_n(m, n));

            c_element_op(acc, acc);
            c1_element_op(c1, c1);
            acc += c1;
            c_m_n(m, n) = ck::type_convert<CDataType>(acc);
        }

    // reduce_mean and reduce_square_mean
    auto reduceSumOpInst = ReduceSumOp{};
    for(int m = 0; m < M; ++m)
    {
        auto mean_acc        = reduceSumOpInst.GetIdentityValue<AccDataType>();
        auto square_mean_acc = reduceSumOpInst.GetIdentityValue<AccDataType>();

        for(int n = 0; n < N; ++n)
        {
            AccDataType c_val        = ck::type_convert<AccDataType>(c_m_n(m, n));
            AccDataType square_c_val = 0;
            UnarySquareElementOp{}(square_c_val, c_val);

            reduceSumOpInst(mean_acc, c_val);
            reduceSumOpInst(square_mean_acc, square_c_val);
        }

        averageOpInst(mean_acc, mean_acc);
        averageOpInst(square_mean_acc, square_mean_acc);
        mean_m(m)       = ck::type_convert<ReduceDataType>(mean_acc);
        meanSquare_m(m) = ck::type_convert<ReduceDataType>(square_mean_acc);
    }

    // LayerNorm
    auto layerNormInst = NormalizeFunctor{};
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            LayerNormOutDataType out_val = 0;
            layerNormInst(out_val, c_m_n(m, n), mean_m(m), meanSquare_m(m), gamma_n(n), beta_n(n));
            out_m_n(m, n) = out_val;
        }
    }
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename BiasDataType,
          typename D0DataType,
          typename ReduceDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename NormalizeDataType>
void DumpGemmLayerNormPerf(float gemm_reduce_time, float normalize_time, int M, int N, int K)
{
    std::size_t gemm_flop     = std::size_t(2) * M * N * K + std::size_t(2) * M * N;
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(CDataType) * M * N + sizeof(BiasDataType) * M * N +
                                sizeof(D0DataType) * M * N + sizeof(ReduceDataType) * M +
                                sizeof(ReduceDataType) * M;

    std::size_t normalize_num_byte = sizeof(CDataType) * M * N + sizeof(ReduceDataType) * M +
                                     sizeof(ReduceDataType) * M + sizeof(GammaDataType) * N +
                                     sizeof(BetaDataType) * N + sizeof(NormalizeDataType) * M * N;

    float tflops               = static_cast<float>(gemm_flop) / 1.E9 / gemm_reduce_time;
    float gemm_gb_per_sec      = gemm_num_byte / 1.E6 / gemm_reduce_time;
    float normalize_gb_per_sec = normalize_num_byte / 1.E6 / normalize_time;

    std::cout << "gemm + reduce_mean + reduce_square_mean Perf: " << gemm_reduce_time << " ms, "
              << tflops << " TFlops, " << gemm_gb_per_sec << " GB/s, " << std::endl;

    std::cout << "5-ary elementwise Perf: " << normalize_time << " ms, " << normalize_gb_per_sec
              << " GB/s, " << std::endl;
}

int main()
{
    // GEMM shape
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA  = 1024;
    ck::index_t StrideB  = 1024;
    ck::index_t StrideC  = 1024;
    ck::index_t StrideD0 = 1024;

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<CDataType> c_m_n(f_host_tensor_descriptor2d(M, N, StrideC, CLayout{}));
    Tensor<BiasDataType> bias_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<D0DataType> c1_m_n(f_host_tensor_descriptor2d(M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> reduceMean_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<ReduceDataType> reduceMeanSquare_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<GammaDataType> gamma_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<BetaDataType> beta_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<LayerNormOutDataType> layerNorm_m_n(
        f_host_tensor_descriptor2d(M, N, StrideC, CLayout{}));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-1, 1});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-1, 1});
    bias_n.GenerateTensorValue(GeneratorTensor_3<BiasDataType>{-1, 1});
    c1_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-5, 5});
    gamma_n.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{-1, 1});
    beta_n.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{-1, 1});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n.mDesc.GetElementSpace());
    DeviceMem bias_device_buf(sizeof(BiasDataType) * bias_n.mDesc.GetElementSpace());
    DeviceMem d0_device_buf(sizeof(D0DataType) * c1_m_n.mDesc.GetElementSpace());
    DeviceMem reduceMean_device_buf(sizeof(ReduceDataType) * reduceMean_m.mDesc.GetElementSpace());
    DeviceMem reduceMeanSquare_device_buf(sizeof(ReduceDataType) *
                                          reduceMeanSquare_m.mDesc.GetElementSpace());
    DeviceMem gamma_device_buf(sizeof(GammaDataType) * gamma_n.mDesc.GetElementSpace());
    DeviceMem beta_device_buf(sizeof(BetaDataType) * beta_n.mDesc.GetElementSpace());
    DeviceMem layerNorm_device_buf(sizeof(LayerNormOutDataType) *
                                   layerNorm_m_n.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    bias_device_buf.ToDevice(bias_n.mData.data());
    d0_device_buf.ToDevice(c1_m_n.mData.data());
    gamma_device_buf.ToDevice(gamma_n.mData.data());
    beta_device_buf.ToDevice(beta_n.mData.data());

    auto a_element_op                     = AElementOp{};
    auto b_element_op                     = BElementOp{};
    auto c_element_op                     = CElementOp{};
    auto d_element_op                     = D0ElementOp{};
    std::array<void*, 3> gemm_element_ops = {&a_element_op, &b_element_op, &c_element_op};

    auto passthrough                            = UnaryIdenticElementOp{};
    auto square                                 = UnarySquareElementOp{};
    auto div                                    = UnaryDivElementOp{N};
    std::array<void*, 2> reduce_in_element_ops  = {&passthrough, &square};
    std::array<void*, 2> reduce_out_element_ops = {&div, &div};

    std::array<void*, 2> p_reduces = {reduceMean_device_buf.GetDeviceBuffer(),
                                      reduceMeanSquare_device_buf.GetDeviceBuffer()};

    // Prepare GEMM, reduce_mean, reduce_mean_square
    auto gemmReduce          = DeviceGemmBiasAddReduceInstance{};
    auto gemmReduce_invoker  = gemmReduce.MakeInvoker();
    auto gemmReduce_argument = gemmReduce.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                                       b_device_buf.GetDeviceBuffer(),
                                                       bias_device_buf.GetDeviceBuffer(),
                                                       {d0_device_buf.GetDeviceBuffer()},
                                                       c_device_buf.GetDeviceBuffer(),
                                                       p_reduces,
                                                       M,
                                                       N,
                                                       K,
                                                       StrideA,
                                                       StrideB,
                                                       StrideC,
                                                       {StrideD0},
                                                       gemm_element_ops,
                                                       {&d_element_op},
                                                       reduce_in_element_ops,
                                                       reduce_out_element_ops);

    if(!gemmReduce.IsSupportedArgument(gemmReduce_argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    reduceMean_device_buf.SetZero();
    reduceMeanSquare_device_buf.SetZero();

    // Prepare LayerNorm
    std::array<const void*, 5> input = {c_device_buf.GetDeviceBuffer(),
                                        reduceMean_device_buf.GetDeviceBuffer(),
                                        reduceMeanSquare_device_buf.GetDeviceBuffer(),
                                        gamma_device_buf.GetDeviceBuffer(),
                                        beta_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {layerNorm_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 2> xyLengths = {M, N};
    std::array<ck::index_t, 2> xyStrides = {StrideC, 1};

    auto normalize         = DeviceNormalizeInstance{};
    auto normalize_invoker = normalize.MakeInvoker();
    auto normalize_argument_ptr =
        normalize.MakeArgumentPointer(xyLengths,
                                      {xyStrides, {1, 0}, {1, 0}, {0, 1}, {0, 1}},
                                      {xyStrides},
                                      input,
                                      output,
                                      NormalizeFunctor{});

    if(!normalize.IsSupportedArgument(normalize_argument_ptr.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device, exiting!");
    }

    // run kernel
    gemmReduce_invoker.Run(gemmReduce_argument, StreamConfig{nullptr, false});
    normalize_invoker.Run(normalize_argument_ptr.get(), StreamConfig{nullptr, false});

    bool pass = true;
    {
        // verification
        Tensor<LayerNormOutDataType> host_layerNorm_m_n(
            f_host_tensor_descriptor2d(M, N, StrideC, CLayout{}));

        host_gemm_layernorm<CDataType, ReduceDataType, ReduceAccDataType>(host_layerNorm_m_n,
                                                                          a_m_k,
                                                                          b_k_n,
                                                                          bias_n,
                                                                          c1_m_n,
                                                                          gamma_n,
                                                                          beta_n,
                                                                          a_element_op,
                                                                          b_element_op,
                                                                          c_element_op,
                                                                          d_element_op,
                                                                          M,
                                                                          N);

        layerNorm_device_buf.FromDevice(layerNorm_m_n.mData.data());
        pass &= ck::utils::check_err(layerNorm_m_n.mData,
                                     host_layerNorm_m_n.mData,
                                     "Error: Incorrect results layerNorm_m_n",
                                     1e-2,
                                     1e-2);
    }

    {
        // evaluate kernel perf
        bool time_kernel = true;

        float gemm_reduce_mean_reduce_square_mean_ave_time =
            gemmReduce_invoker.Run(gemmReduce_argument, StreamConfig{nullptr, time_kernel});
        float normalize_ave_time =
            normalize_invoker.Run(normalize_argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        if(time_kernel)
            DumpGemmLayerNormPerf<ADataType,
                                  BDataType,
                                  CDataType,
                                  BiasDataType,
                                  D0DataType,
                                  ReduceDataType,
                                  GammaDataType,
                                  BetaDataType,
                                  LayerNormOutDataType>(
                gemm_reduce_mean_reduce_square_mean_ave_time, normalize_ave_time, M, N, K);
    }

    return pass ? 0 : 1;
}
