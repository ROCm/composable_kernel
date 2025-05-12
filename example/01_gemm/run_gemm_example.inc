// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <typename DeviceGemm, typename GemmConfig, typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
#endif

    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;

    using ALayout_ = typename GemmConfig::ALayout_;
    using BLayout_ = typename GemmConfig::BLayout_;
    using CLayout_ = typename GemmConfig::CLayout_;
    using ADataType_ = typename GemmConfig::ADataType_;
    using BDataType_ = typename GemmConfig::BDataType_;
    using CDataType_ = typename GemmConfig::CDataType_;

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
            if(stride == -1)
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

    StrideA = f_get_default_stride(M, K, StrideA, ALayout_{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout_{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout_{});

    Tensor<ADataType_> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout_{}));
    Tensor<BDataType_> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout_{}));

    switch(config.init_method)
    {
    case 0:
        ck::utils::FillConstant<ADataType_>{ck::type_convert<ADataType_>(1.f)}(a_m_k);
        ck::utils::FillConstant<BDataType_>{ck::type_convert<BDataType_>(1.f)}(b_k_n);
        break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType_>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType_>{-5.f, 5.f}(b_k_n);
        break;
    case 2:
        ck::utils::FillUniformDistribution<ADataType_>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType_>{-1.f, 1.f}(b_k_n);
        break;
    case 3:
        ck::utils::FillUniformDistributionIntegerValue<ADataType_>{1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType_>{-5.f, 5.f}(b_k_n);
        break;
    case 4:
        ck::utils::FillUniformDistributionIntegerValue<ADataType_>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType_>{1.f, 1.f}(b_k_n);
        break;
    case 5:
        ck::utils::FillUniformDistributionIntegerValue<ADataType_>{-2.f, 2.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType_>{-2.f, 2.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType_>{-0.1f, 0.1f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType_>{-0.1f, 0.1f}(b_k_n);
    }

    Tensor<CDataType_> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout_{}));
    Tensor<CDataType_> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout_{}));
    Tensor<CDataType_> c_m_n_device_ref_result(f_host_tensor_descriptor(M, N, StrideC, CLayout_{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

#ifdef BUILD_INT4_EXAMPLE
    DeviceMem a_m_k_device_buf(sizeof(KernelADataType_) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(KernelBDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(KernelCDataType) *
                               c_m_n_device_result.mDesc.GetElementSpaceSize());

    const Tensor<KernelADataType_> a_m_k_converted(a_m_k);
    const Tensor<KernelBDataType> b_k_n_converted(b_k_n);

    a_m_k_device_buf.ToDevice(a_m_k_converted.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_converted.mData.data());
#else
    DeviceMem a_m_k_device_buf(sizeof(ADataType_) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType_) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType_) * c_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_ref_buf(sizeof(CDataType_) *
                                   c_m_n_device_ref_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
#endif
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm      = DeviceGemm{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

    if constexpr(std::is_same<ProblemType, ProblemSize>::value)
    {
        auto argument = gemm.MakeArgument(
#ifdef BUILD_INT4_EXAMPLE
            static_cast<KernelADataType_*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<KernelBDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<KernelCDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#else
            static_cast<ADataType_*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<BDataType_*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<CDataType_*>(c_m_n_device_buf.GetDeviceBuffer()),
#endif
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
            std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

            return true;
        }

        ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
    }
    else
    {
        // When the Problem Type and Problem Size does not fit.

        std::cerr << gemm.GetTypeString() << ": the instance does not support the problem config."
                  << std::endl;
        return true;
    }

    std::size_t flop = 2_uz * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType_) * M * K + sizeof(BDataType_) * K * N + sizeof(CDataType_) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;

    if((config.do_verification == 1) || (config.do_verification == 3))
    {
        // CPU verification
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        std::cout << "Running verification on CPU." << std::endl;
        ref_invoker.Run(ref_argument);

#ifdef BUILD_INT4_EXAMPLE
        Tensor<CDataType_> c_m_n_device_result_converted(c_m_n_host_result.mDesc);

        c_m_n_device_buf.FromDevice(c_m_n_device_result_converted.mData.data());

        c_m_n_device_result = c_m_n_device_result_converted.CopyAsType<CDataType_>();

        return ck::utils::check_err(c_m_n_device_result_converted, c_m_n_host_result);
#else
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType_>(),
                                     get_atol<CDataType_>());
#endif
    }

    if((config.do_verification == 2) || (config.do_verification == 3))
    {
        // GPU verification
        auto ref_gemm_gpu    = ReferenceGemmInstanceGPU{};
        auto ref_invoker_gpu = ref_gemm_gpu.MakeInvoker();

        auto ref_argument_gpu = ref_gemm_gpu.MakeArgument(
            static_cast<ADataType_*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<BDataType_*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<CDataType_*>(c_m_n_device_ref_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            a_element_op,
            b_element_op,
            c_element_op);

        std::cout << "Running verification on GPU." << std::endl;
        ref_invoker_gpu.Run(ref_argument_gpu, StreamConfig{});

        c_m_n_device_ref_buf.FromDevice(c_m_n_device_ref_result.mData.data());
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_device_ref_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType_>(),
                                     get_atol<CDataType_>());
    }

    return pass == true;
}

struct GemmConfigDefault
{
    using ALayout_ = ALayout;
    using BLayout_ = BLayout;
    using CLayout_ = CLayout;
    using ADataType_ = ADataType;
    using BDataType_ = BDataType;
    using CDataType_ = CDataType;
};

bool run_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm<DeviceGemmInstance, GemmConfigDefault>(problem_size, config);
}

template<typename DeviceGemm, typename GemmConfig>
bool run_gemm_example_with_instance(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm<DeviceGemm, GemmConfig>(problem_size, config);
}
