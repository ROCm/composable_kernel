#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_c_permute_xdl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

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

// static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
// static constexpr auto MNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;
static constexpr auto MNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmCPermuteXdl
//######| ALayout| BLayout| AData| BData| CData| AccData|           A|           B|           C|          GEMM|      Num| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|     CShuffle|    CShuffle| CBlockTransferClusterLengths|   CBlockTransfer|
//######|        |        |  Type|  Type|  Type|    Type| Elementwise| Elementwise| Elementwise|Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|  MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl|  ScalarPerVector|
//######|        |        |      |      |      |        |   Operation|   Operation|   Operation|              |         |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |   PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|    _NWaveNPerXdl|
//######|        |        |      |      |      |        |            |            |            |              |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |             |            |                             |                 |
//      <     Row,     Col,   F16,   F16,   F16,     F32, PassThrough, PassThrough, PassThrough,     MNPadding,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,            1,           1,               S<1, 32, 1, 8>,                8>;
        <     Row,     Col,   F16,   F16,   F16,     F32, PassThrough, PassThrough, PassThrough,     MNKPadding,        1,   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,            1,           1,               S<1, 32, 1, 8>,                8>;
// clang-format on

using ReferenceBatchedGemmInstance = ck::tensor_operation::host::
    ReferenceBatchedGemm<ADataType, BDataType, CDataType, AElementOp, BElementOp, CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    const int M = 88;
    const int N = 64;
    const int K = 88;

    const int stride_A = K;
    const int stride_B = K;

    const int G0 = 1024;
    const int G1 = 10;

    const int batch_count = G0 * G1;

    // output layout - [G0, M, G1, N]
    const int stride_G0 = M * G1 * N;
    const int stride_G1 = N;
    const int stride_M  = G1 * N;
    const int stride_N  = 1;

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
    ck::tensor_operation::device::BatchedGemmCPermuteDesc batched_gemm_c_permute_desc{
        G0, G1, M, N, stride_G0, stride_G1, stride_M, stride_N};

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

    auto f_host_c_tensor_descriptor = [](std::size_t G0_,
                                         std::size_t G1_,
                                         std::size_t M_,
                                         std::size_t N_,
                                         std::size_t stride_G0_,
                                         std::size_t stride_G1_,
                                         std::size_t stride_M_,
                                         std::size_t stride_N_) {
        return HostTensorDescriptor(
            std::vector<std::size_t>({G0_, G1_, M_, N_}),
            std::vector<std::size_t>({stride_G0_, stride_G1_, stride_M_, stride_N_}));
    };

    Tensor<CDataType> c_g0_g1_m_n_host_result(
        f_host_c_tensor_descriptor(G0, G1, M, N, stride_G0, stride_G1, stride_M, stride_N));

    Tensor<CDataType> c_g0_g1_m_n_device_result(
        f_host_c_tensor_descriptor(G0, G1, M, N, stride_G0, stride_G1, stride_M, stride_N));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_g0_g1_m_n: " << c_g0_g1_m_n_host_result.mDesc << std::endl;

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
    DeviceMem c_device_buf(sizeof(CDataType) * c_g0_g1_m_n_device_result.mDesc.GetElementSpace());

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
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      batched_gemm_c_permute_desc,
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
        c_device_buf.FromDevice(c_g0_g1_m_n_device_result.mData.data());

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        Tensor<CDataType> c_g_m_n_host_result = HostTensorDescriptor(
            std::vector<std::size_t>({batch_count, M, N}), std::vector<std::size_t>({M * N, N, 1}));

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, c_g_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        for(int g0 = 0; g0 < G0; g0++)
        {
            for(int g1 = 0; g1 < G1; g1++)
            {
                for(int m = 0; m < M; m++)
                {
                    for(int n = 0; n < N; n++)
                    {
                        int g                                 = g0 * G1 + g1;
                        c_g0_g1_m_n_host_result(g0, g1, m, n) = c_g_m_n_host_result(g, m, n);
                    }
                }
            }
        }

        pass = ck::utils::check_err(c_g0_g1_m_n_host_result.mData,
                                    c_g0_g1_m_n_device_result.mData,
                                    "Error: Incorrect results c");
    }

    return pass ? 0 : 1;
}
