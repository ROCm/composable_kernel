// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
using CDataType        = ck::bhalf_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

#if 1
static const uint32_t AB_K1 = 8;

// clang-format off
template <bool UseDataCachePrefetch>
using DeviceGemmV3Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        128,
        256, 256,
        64, AB_K1, AB_K1,
        16,   16,
        8,    16,
        S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, AB_K1, AB_K1, 0,
        S<8, 16, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, AB_K1, AB_K1, 0,
        2, 4, S<1, 8, 1, 16>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3,
        CDataType,
        CDataType,
        false,
        false,
        0,
        UseDataCachePrefetch>;
// clang-format on
#else
// prefetch is faster on these params
// clang-format off
template <bool UseDataCachePrefetch>
using DeviceGemmV3Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        256,
        128, 128, 
        64, 8, 8,
        16,   16,
        4,    4,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 2, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3,
        CDataType,
        CDataType,
        false,
        false,
        0,
        UseDataCachePrefetch>;
// clang-format on
#endif

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

template <typename GemmInstanceType, typename ProblemType>
std::pair<bool, float> run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
            if(stride == -1 || stride == 0)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());

    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm      = GemmInstanceType{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      KBatch,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return std::make_pair(true, ave_time);
    }

    bool pass = true;
    if((config.do_verification == 1) || (config.do_verification == 3))
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        ave_time = invoker.Run(argument, StreamConfig{nullptr, false, 1});
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());
    }

    if(config.time_kernel)
    {
        ave_time =
            invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 0, 50, 100, true, 4});

        std::size_t flop = 2_uz * M * N * K;
        std::size_t num_btype =
            sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }
    return std::make_pair(pass, ave_time);
}

bool parse_cmd_args(int argc,
                    char* argv[],
                    ProblemSizeSplitK& problem_size,
                    ExecutionConfig& config,
                    bool& compareWithNonDataCachePrefetchImpl)
{
    compareWithNonDataCachePrefetchImpl = false;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4 || argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        if(argc >= 10)
        {

            problem_size.M = std::stoi(argv[4]);
            problem_size.N = std::stoi(argv[5]);
            problem_size.K = std::stoi(argv[6]);

            problem_size.StrideA = std::stoi(argv[7]);
            problem_size.StrideB = std::stoi(argv[8]);
            problem_size.StrideC = std::stoi(argv[9]);

            if(argc >= 11)
            {
                problem_size.KBatch = std::stoi(argv[10]);
                if(argc > 12)
                {
                    compareWithNonDataCachePrefetchImpl = std::stoi(argv[11]);
                }
            }
        }
    }
    else
    {
        std::cerr
            << "arg1: verification (0=no, 1=CPU, 2=GPU, 3=CPU and GPU)" << std::endl
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)" << std::endl
            << "arg3: time kernel (0=no, 1=yes)" << std::endl
            << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC (default: -1 or 0)"
            << std::endl
            << "arg10: KBatch" << std::endl
            << "arg11: compareWithNonDataCachePrefetchImpl (0=no, 1=yes)" << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;
    bool compareWithNonDataCachePrefetchImpl;

    if(!parse_cmd_args(argc, argv, problem_size, config, compareWithNonDataCachePrefetchImpl))
    {
        return 1;
    }

    auto [pass, ave_time] = run_gemm<DeviceGemmV3Instance<true>>(problem_size, config);

    if(compareWithNonDataCachePrefetchImpl)
    {
        auto [pass2, ave_time2] = run_gemm<DeviceGemmV3Instance<false>>(problem_size, config);
        std::cout << "DataCache Prefetching enabled ave_time: " << ave_time << " ms" << std::endl;
        std::cout << "DataCache Prefetching disabled ave_time: " << ave_time2 << " ms" << std::endl;
        float speedup = ave_time2 / ave_time;
        std::cout << "On average kernel with DataCache prefetching is " << speedup
                  << " times faster than without DataCache prefetching." << std::endl;

        if(speedup < 1.0f)
            std::cout << "WARNING: Kernel with DataCache prefetching is slower!" << std::endl;
    }

    return pass ? 0 : 1;
}
