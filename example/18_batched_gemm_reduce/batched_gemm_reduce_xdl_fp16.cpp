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
#include "device_batched_gemm_reduce_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "reduction_operator.hpp"
#include "reference_batched_gemm.hpp"
#include "gemm_specialization.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType         = F16;
using BDataType         = F16;
using CDataType         = F16;
using ReduceAccDataType = F32;
using DDataType         = F32;
using DPtrsGlobal       = ck::Tuple<DDataType*, DDataType*>;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp  = ck::tensor_operation::element_wise::PassThrough;
using BElementOp  = ck::tensor_operation::element_wise::PassThrough;
using CElementOp  = ck::tensor_operation::element_wise::PassThrough;
using D0ReduceOp  = ck::reduce::Add<ReduceAccDataType>;
using D1ReduceOp  = ck::reduce::Add<ReduceAccDataType>;
using DxsReduceOp = ck::Tuple<D0ReduceOp, D1ReduceOp>;

using UnaryIdenticElementOp =
    ck::tensor_operation::element_wise::UnaryIdentic<ReduceAccDataType, ReduceAccDataType, false>;
using UnarySquareElementOp =
    ck::tensor_operation::element_wise::UnarySquare<ReduceAccDataType, ReduceAccDataType, false>;
using DxsInElementOp  = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
using DxsOutElementOp = ck::Tuple<UnaryIdenticElementOp, UnaryIdenticElementOp>;

using DGlobalMemOp =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicAdd,
                                          ck::InMemoryDataOperationEnum::AtomicAdd>;

static constexpr auto GemmSpecialization =
    ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceBatchedGemmReduceInstance = ck::tensor_operation::device::DeviceBatchedGemmReduce_Xdl_CShuffle
//######| ALayout| BLayout| CLayout|AData| BData| CData|  GemmAcc| CShuffle| ReduceAcc|         DData|           A|           B|           C|         Dxs|     DxsInEleOp|     DxsAccEleOp|            D|               GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|              CReduce| CReduceThreadLds2VGprCopy| CReduceThreadVgpr2GlobalCopy|
//######|        |        |        | Type|  Type|  Type| DataType| DataType|  DataType|    Type Tuple| Elementwise| Elementwise| Elementwise|      Reduce|               |                |   MemoryData|     Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN| MXdlPerWave| NXdlPerWave|            _MBlock_MPerBlock| ScalarPerVector| ThreadClusterLengths|     SrcDstScalarPerVector|        SrcDstScalarPerVector|
//######|        |        |        |     |      |      |         |         |          |              |   Operation|   Operation|   Operation|   Operation|               |                |    Operation|                   |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|            _NBlock_NPerBlock|      _NPerBlock| _MPerBlock_NPerBlock|                _NPerBlock|                   _MPerBlock|
//######|        |        |        |     |      |      |         |         |          |              |            |            |            |            |               |                |             |                   |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                     |                          |                             |
        <     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,       F32,   DPtrsGlobal,  AElementOp,  BElementOp,  CElementOp, DxsReduceOp, DxsInElementOp, DxsOutElementOp, DGlobalMemOp, GemmSpecialization,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8,             S<64, 4>,                         4,                            1>;
// clang-format on

using ReferenceBatchedGemmInstance = ck::tensor_operation::host::
    ReferenceBatchedGemm<ADataType, BDataType, CDataType, AElementOp, BElementOp, CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 2048;
    ck::index_t N = 1920;
    ck::index_t K = 2048;

    ck::index_t StrideA = 2048;
    ck::index_t StrideB = 2048;
    ck::index_t StrideC = 1920;

    ck::index_t BatchCount = 4;

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
    else if(argc == 11)
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

        BatchCount = std::stoi(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4 to 10: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC, BatchCount\n");
        exit(0);
    }

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       auto layout) {
        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({row * stride, stride, 1}));
        }
        else
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({col * stride, 1, stride}));
        }
    };

    Tensor<ADataType> a_g_m_k(f_host_tensor_descriptor(BatchCount, M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_g_k_n(f_host_tensor_descriptor(BatchCount, K, N, StrideB, BLayout{}));

    Tensor<CDataType> c_g_m_n_host_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    Tensor<DDataType> d0_g_m_host_result(HostTensorDescriptor(std::vector<std::size_t>(
        {static_cast<std::size_t>(BatchCount), static_cast<std::size_t>(M)})));
    Tensor<DDataType> d1_g_m_host_result(HostTensorDescriptor(std::vector<std::size_t>(
        {static_cast<std::size_t>(BatchCount), static_cast<std::size_t>(M)})));

    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    Tensor<DDataType> d0_g_m_device_result(HostTensorDescriptor(std::vector<std::size_t>(
        {static_cast<std::size_t>(BatchCount), static_cast<std::size_t>(M)})));
    Tensor<DDataType> d1_g_m_device_result(HostTensorDescriptor(std::vector<std::size_t>(
        {static_cast<std::size_t>(BatchCount), static_cast<std::size_t>(M)})));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m_n_host_result.mDesc << std::endl;
    std::cout << "d0_g_m: " << d0_g_m_host_result.mDesc << std::endl;
    std::cout << "d1_g_m: " << d1_g_m_host_result.mDesc << std::endl;

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
    DeviceMem c_device_buf(sizeof(CDataType) * c_g_m_n_device_result.mDesc.GetElementSpace());
    DeviceMem d0_device_buf(sizeof(DDataType) * d0_g_m_device_result.mDesc.GetElementSpace());
    DeviceMem d1_device_buf(sizeof(DDataType) * d1_g_m_device_result.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};
    auto dxs_global   = ck::make_tuple(static_cast<DDataType*>(d0_device_buf.GetDeviceBuffer()),
                                     static_cast<DDataType*>(d1_device_buf.GetDeviceBuffer()));

    // do GEMM
    auto batched_gemm = DeviceBatchedGemmReduceInstance{};
    auto invoker      = batched_gemm.MakeInvoker();
    auto argument =
        batched_gemm.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                  static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                  static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                  dxs_global,
                                  M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  StrideC,
                                  a_element_op,
                                  b_element_op,
                                  c_element_op,
                                  DxsInElementOp{},
                                  DxsOutElementOp{},
                                  BatchCount);

    if(!batched_gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    // init DO, D1 to 0
    d0_device_buf.SetZero();
    d1_device_buf.SetZero();

    // if time_kernel == true, kernel will run multiple times. This kernel use atomic-add so result
    // will not be correct. need to set time_kernel = false for correctness test
    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = std::size_t(2) * BatchCount * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * BatchCount * M * K +
                            sizeof(BDataType) * BatchCount * K * N +
                            sizeof(CDataType) * BatchCount * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << batched_gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_device_buf.FromDevice(c_g_m_n_device_result.mData.data());
        d0_device_buf.FromDevice(d0_g_m_device_result.mData.data());
        d1_device_buf.FromDevice(d1_g_m_device_result.mData.data());

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, c_g_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        auto d0_reduce_op = D0ReduceOp{};
        auto d1_reduce_op = D1ReduceOp{};

        for(int batch = 0; batch < BatchCount; ++batch)
        {
            for(int m = 0; m < M; ++m)
            {
                float d0_acc = d0_reduce_op.GetIdentityValue();
                float d1_acc = d1_reduce_op.GetIdentityValue();

                for(int n = 0; n < N; ++n)
                {
                    float c_val  = ck::type_convert<float>(c_g_m_n_host_result(batch, m, n));
                    float d0_val = 0;
                    float d1_val = 0;

                    UnaryIdenticElementOp{}(d0_val, c_val);
                    UnarySquareElementOp{}(d1_val, c_val);
                    d0_reduce_op(d0_acc, d0_val);
                    d1_reduce_op(d1_acc, d1_val);
                }

                d0_g_m_host_result(batch, m) = ck::type_convert<DDataType>(d0_acc);
                d1_g_m_host_result(batch, m) = ck::type_convert<DDataType>(d1_acc);
            }
        }

        pass = ck::utils::check_err(c_g_m_n_host_result.mData,
                                    c_g_m_n_device_result.mData,
                                    "Error: Incorrect results c") &&
               ck::utils::check_err(d0_g_m_device_result.mData,
                                    d0_g_m_host_result.mData,
                                    "Error: Incorrect results! D0",
                                    1e-4,
                                    1e-5) &&
               ck::utils::check_err(d1_g_m_device_result.mData,
                                    d1_g_m_host_result.mData,
                                    "Error: Incorrect results! D1",
                                    1e-3,
                                    1e-5);
    }

    return pass ? 0 : 1;
}
