#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "check_err.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_batched_gemm_transpose_xdl.hpp"
#include "element_wise_operation.hpp"
#include "reference_batched_gemm_transpose.hpp"
#include "gemm_specialization.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmTransposeXdl
//######| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
//######|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
//######|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
//######|      |      |      |        |        |        |        |            |            |            |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
        <   F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,   256,   256,   128,     4,  8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1>;
// clang-format on

using ReferenceBatchedGemmTransposeInstance =
    ck::tensor_operation::host::ReferenceBatchedGemmTranspose<ADataType,
                                                              BDataType,
                                                              CDataType,
                                                              AElementOp,
                                                              BElementOp,
                                                              CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    const int M0 = rand() % 4 + 1;
    const int M1 = 256;
    const int N0 = rand() % 4 + 1;
    const int N1 = 256;

    const int M = M0 * N1;
    const int N = N0 * N1;

    const int K = 128 * (rand() % 4 + 1);

    const int stride_A = K;
    const int stride_B = K;

    // output layout [M0, N0, M1, N1]
    const int stride_M0 = N1 * M1 * N0;
    const int stride_M1 = N1;
    const int stride_N0 = N1 * M1;
    const int stride_N1 = 1;

    int batch_count = rand() % 16 + 1;

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

    // GEMM shape
    ck::tensor_operation::device::GemmTransposeDesc gemm_transpose_desc{
        M, N, K, stride_A, stride_B, M0, M1, N0, N1, stride_M0, stride_M1, stride_N0, stride_N1};

    auto f_host_tensor_descriptor = [](std::size_t batch_count_,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       auto layout) {
        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count_, row, col}),
                                        std::vector<std::size_t>({row * stride, stride, 1}));
        }
        else
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count_, row, col}),
                                        std::vector<std::size_t>({col * stride, 1, stride}));
        }
    };

    Tensor<ADataType> a_g_m_k(f_host_tensor_descriptor(batch_count, M, K, stride_A, ALayout{}));
    Tensor<BDataType> b_g_k_n(f_host_tensor_descriptor(batch_count, K, N, stride_B, BLayout{}));

    auto f_host_c_tensor_descriptor = [](std::size_t batch_count_,
                                         std::size_t M0_,
                                         std::size_t M1_,
                                         std::size_t N0_,
                                         std::size_t N1_,
                                         std::size_t StrideM0_,
                                         std::size_t StrideM1_,
                                         std::size_t StrideN0_,
                                         std::size_t StrideN1_) {
        return HostTensorDescriptor(
            std::vector<std::size_t>({batch_count_, M0_, M1_, N0_, N1_}),
            std::vector<std::size_t>(
                {M0_ * M1_ * N0_ * N1_, StrideM0_, StrideM1_, StrideN0_, StrideN1_}));
    };

    Tensor<CDataType> c_g_m0_m1_n0_n1_host_result(f_host_c_tensor_descriptor(
        batch_count, M0, M1, N0, N1, stride_M0, stride_M1, stride_N0, stride_N1));

    Tensor<CDataType> c_g_m0_m1_n0_n1_device_result(f_host_c_tensor_descriptor(
        batch_count, M0, M1, N0, N1, stride_M0, stride_M1, stride_N0, stride_N1));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m0_m1_n0_n1_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) *
                           c_g_m0_m1_n0_n1_device_result.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    // do GEMM
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                      gemm_transpose_desc,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op,
                                      batch_count);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = std::size_t(2) * batch_count * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * batch_count * M * K +
                            sizeof(BDataType) * batch_count * K * N +
                            sizeof(CDataType) * batch_count * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;

    if(do_verification)
    {
        c_device_buf.FromDevice(c_g_m0_m1_n0_n1_device_result.mData.data());

        auto ref_batched_gemm = ReferenceBatchedGemmTransposeInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        auto ref_argument = ref_batched_gemm.MakeArgument(a_g_m_k,
                                                          b_g_k_n,
                                                          c_g_m0_m1_n0_n1_host_result,
                                                          a_element_op,
                                                          b_element_op,
                                                          c_element_op);

        ref_invoker.Run(ref_argument);

        pass = ck::utils::check_err(c_g_m0_m1_n0_n1_host_result.mData,
                                    c_g_m0_m1_n0_n1_device_result.mData,
                                    "Error: Incorrect results c");
    }

    return pass ? 0 : 1;
}
