#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "device_gemm_xdl_layernorm_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "reduction_operator.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType = F16;
using BDataType = F16;
using CDataType = F16;
using AccDataType = F32;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp  = ck::tensor_operation::element_wise::PassThrough;
using BElementOp  = ck::tensor_operation::element_wise::PassThrough;
using CElementOp  = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmLayerNorm_Xdl_CShuffle
//######| ALayout| BLayout| CLayout|AData| BData| CData|     GemmAcc|    CShuffle|   ReduceAcc|           A|           B|           C|               GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|              CReduce| CReduceThreadLds2VGprCopy| CReduceThreadVgpr2GlobalCopy|
//######|        |        |        | Type|  Type|  Type|    DataType|    DataType|    DataType| Elementwise| Elementwise| Elementwise|     Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN| MXdlPerWave| NXdlPerWave|            _MBlock_MPerBlock| ScalarPerVector| ThreadClusterLengths|     SrcDstScalarPerVector|        SrcDstScalarPerVector|
//######|        |        |        |     |      |      |            |            |            |   Operation|   Operation|   Operation|                   |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|            _NBlock_NPerBlock|      _NPerBlock| _MPerBlock_NPerBlock|                _NPerBlock|                   _MPerBlock|
//######|        |        |        |     |      |      |            |            |            |            |            |            |                   |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                     |                          |                             |
        <     Row,     Col,     Row,  F16,   F16,   F16, AccDataType, AccDataType, AccDataType,  AElementOp,  BElementOp,  CElementOp,        GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           2,               S<1, 32, 1, 8>,               8,             S<64, 4>,                         4,                            1>;
// clang-format on

// D = Layernorm(acc + broadcast(bias)) * broadcast(gamma) + broadcast(beta)
template <typename InDataType, typename OutDataType>
void Layernorm(Tensor<OutDataType>& result,
               const Tensor<InDataType>& acc,   // MxN
               const Tensor<InDataType>& bias,  // 1xN
               const Tensor<InDataType>& gamma, // 1xN
               const Tensor<InDataType>& beta,  // 1xN
               const InDataType epsilon = 1e-5)
{
    assert(acc.mDesc.GetLengths()[1] == bias.mDesc.GetLengths()[0] &&
           acc.mDesc.GetLengths()[1] == gamma.mDesc.GetLengths()[0] &&
           acc.mDesc.GetLengths()[1] == beta.mDesc.GetLengths()[0]);

    size_t M = acc.mDesc.GetLengths()[0];
    size_t N = acc.mDesc.GetLengths()[1];

    Tensor<InDataType> avg_acc_sq(HostTensorDescriptor(std::vector<size_t>({M})));
    Tensor<InDataType> avg_acc(HostTensorDescriptor(std::vector<size_t>({M})));
    Tensor<InDataType> acc_layernorm(acc.mDesc);

    // add bias
    acc_layernorm.ForEach([&](auto& self, auto idx) {
        self(idx[0], idx[1]) = acc(idx[0], idx[1]) + bias(idx[1]);
    });

    // reduce N dim
    for(size_t i = 0; i < M; i++)
    {
        InDataType sum_acc_sq = 0;
        InDataType sum_acc    = 0;
        for(size_t j = 0; j < N; j++)
        {
            sum_acc_sq += acc_layernorm(i, j) * acc_layernorm(i, j);
            sum_acc += acc_layernorm(i, j);
        }
        avg_acc_sq(i) = sum_acc_sq / N;
        avg_acc(i)    = sum_acc / N;
        // std::cout << "avg_acc_(" << i << ") =" << avg_acc(i) << std::endl;
        // std::cout << "avg_acc_sq_(" << i << ") =" << avg_acc_sq(i) << std::endl;
    }

    // normalize
    acc_layernorm.ForEach([&](auto& self, auto idx) {
        self(idx[0], idx[1]) =
            (self(idx[0], idx[1]) - avg_acc(idx[0])) /
            sqrt(avg_acc_sq(idx[0]) - avg_acc(idx[0]) * avg_acc(idx[0]) + epsilon);
    });

    // affine
    acc_layernorm.ForEach([&](auto& self, auto idx) {
        self(idx[0], idx[1]) = self(idx[0], idx[1]) * gamma(idx[1]) + beta(idx[1]);
    });

    // cast
    result = acc_layernorm.template CopyAsType<OutDataType>();
}

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 128;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 128;

    if(argc == 1)
    {
        // do nothing
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideC = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
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

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<AccDataType> acc_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<AccDataType> c0_n_bias(HostTensorDescriptor(std::vector<size_t>({size_t(N)})));
    Tensor<AccDataType> c0_n_gamma(HostTensorDescriptor(std::vector<size_t>({size_t(N)})));
    Tensor<AccDataType> c0_n_beta(HostTensorDescriptor(std::vector<size_t>({size_t(N)})));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "c0_n_bias: " << c0_n_bias.mDesc << std::endl;
    std::cout << "c0_n_gamma: " << c0_n_gamma.mDesc << std::endl;
    std::cout << "c0_n_beta: " << c0_n_beta.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_Sequential<0>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
    }

    // TODO ANT: test other init
    c_m_n_host_result.GenerateTensorValue(GeneratorTensor_1<CDataType>{0});
    acc_m_n_host_result.GenerateTensorValue(GeneratorTensor_1<AccDataType>{0});
    c0_n_bias.GenerateTensorValue(GeneratorTensor_Sequential<0>{});
    c0_n_gamma.GenerateTensorValue(GeneratorTensor_1<AccDataType>{2});
    c0_n_beta.GenerateTensorValue(GeneratorTensor_1<AccDataType>{2});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());
    DeviceMem c0_bias_buf(sizeof(AccDataType) * c0_n_bias.mDesc.GetElementSpace());
    DeviceMem c0_gamma_buf(sizeof(AccDataType) * c0_n_gamma.mDesc.GetElementSpace());
    DeviceMem c0_beta_buf(sizeof(AccDataType) * c0_n_beta.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    c0_bias_buf.ToDevice(c0_n_bias.mData.data());
    c0_gamma_buf.ToDevice(c0_n_gamma.mData.data());
    c0_beta_buf.ToDevice(c0_n_beta.mData.data());

    auto a_element_op  = AElementOp{};
    auto b_element_op  = BElementOp{};
    auto c_element_op  = CElementOp{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                      static_cast<AccDataType*>(c0_bias_buf.GetDeviceBuffer()),
                                      static_cast<AccDataType*>(c0_gamma_buf.GetDeviceBuffer()),
                                      static_cast<AccDataType*>(c0_beta_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_device_buf.FromDevice(c_m_n_device_result.mData.data());

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, acc_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        Layernorm(c_m_n_host_result, acc_m_n_host_result, c0_n_bias, c0_n_gamma, c0_n_beta);

        pass &= ck::utils::check_err(
            c_m_n_device_result.mData, c_m_n_host_result.mData, "Error: Incorrect results c");

        // if (!pass)
        // {
        //     LogRangeAsType<float>(std::cout << "c_host: ", c_m_n_host_result.mData, ",")
        //                 << std::endl;
        //     LogRangeAsType<float>(std::cout << "c_device: ", c_m_n_device_result.mData, ",")
        //                 << std::endl;
        // }
    }
    return pass ? 0 : 1;
}
