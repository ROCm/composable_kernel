// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = float;
using BDataType   = float;
using CDataType   = float;
using AccDataType = float;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   4,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>,              4>;
// clang-format on

// hardcoded for NumDimM == NumDimN == NumDimK == 2
template <ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::enable_if_t<NumDimM == 2 && NumDimN == 2 && NumDimK == 2, bool> = false>
struct ReferenceContraction_M2_N2_K2 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_ms_ks,
                 const Tensor<BDataType>& b_ks_ns,
                 Tensor<CDataType>& c_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_ms_ks_{a_ms_ks},
              b_ks_ns_{b_ks_ns},
              c_ms_ns_{c_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ADataType>& a_ms_ks_;
        const Tensor<BDataType>& b_ks_ns_;
        Tensor<CDataType>& c_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceContraction_M2_N2_K2::Argument;

        float Run(const Argument& arg)
        {
            auto f_ms_ns = [&](auto m0, auto m1, auto n0, auto n1) {
                const int K0 = arg.a_ms_ks_.mDesc.GetLengths()[2];
                const int K1 = arg.a_ms_ks_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(int k0 = 0; k0 < K0; ++k0)
                {
                    for(int k1 = 0; k1 < K1; ++k1)
                    {
                        AccDataType v_a;
                        AccDataType v_b;

                        arg.a_element_op_(
                            v_a, static_cast<const AccDataType>(arg.a_ms_ks_(m0, m1, k0, k1)));
                        arg.b_element_op_(
                            v_b, static_cast<const AccDataType>(arg.b_ks_ns_(k0, k1, n0, n1)));

                        v_acc += v_a * v_b;
                    }
                }

                AccDataType v_c;

                arg.c_element_op_(v_c, v_acc);

                arg.c_ms_ns_(m0, m1, n0, n1) = v_c;
            };

            make_ParallelTensorFunctor(f_ms_ns,
                                       arg.c_ms_ns_.mDesc.GetLengths()[0],
                                       arg.c_ms_ns_.mDesc.GetLengths()[1],
                                       arg.c_ms_ns_.mDesc.GetLengths()[2],
                                       arg.c_ms_ns_.mDesc.GetLengths()[3])(
                std::thread::hardware_concurrency());

            return 0;
        }

        float Run(const ck::tensor_operation::device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const ck::tensor_operation::device::BaseArgument*) override
    {
        return true;
    }

    static auto MakeArgument(const Tensor<ADataType>& a_ms_ks,
                             const Tensor<BDataType>& b_ks_ns,
                             Tensor<CDataType>& c_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{a_ms_ks, b_ks_ns, c_ms_ns, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<ck::tensor_operation::device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceContraction_M2_N2_K2"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

using ReferenceOpInstance = ReferenceContraction_M2_N2_K2<NumDimM,
                                                          NumDimN,
                                                          NumDimK,
                                                          ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          AccDataType,
                                                          AElementOp,
                                                          BElementOp,
                                                          CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

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
        printf("arg3: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    // A[M0, M1, K0, K1]
    std::vector<ck::index_t> a_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a_ms_ks_strides{524288, 4096, 128, 1};
    // B[K0, K1, N0, N1]
    std::vector<ck::index_t> b_ks_ns_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b_ks_ns_strides{128, 1, 524288, 4096};
    // C[M0, M1, N0, N1]
    std::vector<ck::index_t> c_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> c_ms_ns_strides{524288, 4096, 128, 1};

    Tensor<ADataType> a_ms_ks(
        std::vector<std::size_t>(a_ms_ks_lengths.begin(), a_ms_ks_lengths.end()),
        std::vector<std::size_t>(a_ms_ks_strides.begin(), a_ms_ks_strides.end()));
    Tensor<BDataType> b_ks_ns(
        std::vector<std::size_t>(b_ks_ns_lengths.begin(), b_ks_ns_lengths.end()),
        std::vector<std::size_t>(b_ks_ns_strides.begin(), b_ks_ns_strides.end()));
    Tensor<CDataType> c_ms_ns_host_result(
        std::vector<std::size_t>(c_ms_ns_lengths.begin(), c_ms_ns_lengths.end()),
        std::vector<std::size_t>(c_ms_ns_strides.begin(), c_ms_ns_strides.end()));
    Tensor<CDataType> c_ms_ns_device_result(
        std::vector<std::size_t>(c_ms_ns_lengths.begin(), c_ms_ns_lengths.end()),
        std::vector<std::size_t>(c_ms_ns_strides.begin(), c_ms_ns_strides.end()));

    std::cout << "a_ms_ks: " << a_ms_ks.mDesc << std::endl;
    std::cout << "b_ks_ns: " << b_ks_ns.mDesc << std::endl;
    std::cout << "c_ms_ns: " << c_ms_ns_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_ks_ns.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    case 2:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_ks_ns.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    default:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<0>{});
        b_ks_ns.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
    }

    DeviceMem a_ms_ks_device_buf(sizeof(ADataType) * a_ms_ks.mDesc.GetElementSpace());
    DeviceMem b_ks_ns_device_buf(sizeof(BDataType) * b_ks_ns.mDesc.GetElementSpace());
    DeviceMem c_ms_ns_device_buf(sizeof(CDataType) * c_ms_ns_device_result.mDesc.GetElementSpace());

    a_ms_ks_device_buf.ToDevice(a_ms_ks.mData.data());
    b_ks_ns_device_buf.ToDevice(b_ks_ns.mData.data());

    // set zero
    c_ms_ns_device_buf.SetZero();

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // device operation
    auto op       = DeviceOpInstance{};
    auto invoker  = op.MakeInvoker();
    auto argument = op.MakeArgument(static_cast<ADataType*>(a_ms_ks_device_buf.GetDeviceBuffer()),
                                    static_cast<BDataType*>(b_ks_ns_device_buf.GetDeviceBuffer()),
                                    static_cast<CDataType*>(c_ms_ns_device_buf.GetDeviceBuffer()),
                                    a_ms_ks_lengths,
                                    a_ms_ks_strides,
                                    b_ks_ns_lengths,
                                    b_ks_ns_strides,
                                    c_ms_ns_lengths,
                                    c_ms_ns_strides,
                                    a_element_op,
                                    b_element_op,
                                    c_element_op);

    if(!op.IsSupportedArgument(argument))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    ck::index_t M = std::accumulate(c_ms_ns_lengths.begin(),
                                    c_ms_ns_lengths.begin() + NumDimM,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    ck::index_t N = std::accumulate(c_ms_ns_lengths.begin() + NumDimM,
                                    c_ms_ns_lengths.begin() + NumDimM + NumDimN,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    ck::index_t K = std::accumulate(a_ms_ks_lengths.begin() + NumDimM,
                                    a_ms_ks_lengths.begin() + NumDimM + NumDimK,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << op.GetTypeString() << std::endl;

    c_ms_ns_device_buf.FromDevice(c_ms_ns_device_result.mData.data());

    if(do_verification)
    {
        auto ref_gemm    = ReferenceOpInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_ms_ks, b_ks_ns, c_ms_ns_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        return ck::utils::check_err(c_ms_ns_device_result.mData, c_ms_ns_host_result.mData) ? 0 : 1;
    }

    return 0;
}
