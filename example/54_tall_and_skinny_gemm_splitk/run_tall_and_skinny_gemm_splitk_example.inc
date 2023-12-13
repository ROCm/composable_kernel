// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

bool run_tall_and_skinny_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
#endif

    using namespace ck::literals;

    auto& [M, N, K, StrideA, StrideB, StrideC, k_batch] = problem_size; // //

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

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

#ifdef BUILD_INT4_EXAMPLE
    DeviceMem a_m_k_device_buf(sizeof(KernelADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(KernelBDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(KernelCDataType) *
                               c_m_n_device_result.mDesc.GetElementSpaceSize());

    const Tensor<KernelADataType> a_m_k_converted(a_m_k);
    const Tensor<KernelBDataType> b_k_n_converted(b_k_n);

    a_m_k_device_buf.ToDevice(a_m_k_converted.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_converted.mData.data());
#else
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
#endif

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto tsmm     = DeviceTSMMInstance{};
    auto invoker  = tsmm.MakeInvoker();
    auto argument = tsmm.MakeArgument(
#ifdef BUILD_INT4_EXAMPLE
        static_cast<KernelADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<KernelBDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
        static_cast<KernelCDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#else
        static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#endif
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideC,
        a_element_op,
        b_element_op,
        c_element_op,
        k_batch); // //

    // //
    if(!tsmm.IsSupportedArgument(argument))
    {
        std::cerr << tsmm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    c_m_n_device_buf.SetZero();

    if(config.do_verification)
    {
        invoker.Run(argument, StreamConfig{nullptr, false}); // Run prior to verification
        auto ref_tsmm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_tsmm.MakeInvoker();

        auto ref_argument = ref_tsmm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

#ifdef BUILD_INT4_EXAMPLE
        Tensor<CDataType> c_m_n_device_result_converted(c_m_n_host_result.mDesc);

        c_m_n_device_buf.FromDevice(c_m_n_device_result_converted.mData.data());

        c_m_n_device_result = c_m_n_device_result_converted.CopyAsType<CDataType>();

#else
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

#endif
    }

    float ave_time = invoker.Run(
        argument, StreamConfig{nullptr, config.time_kernel}); // Run to measure performance

    std::size_t flop = 2_uz * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << tsmm.GetTypeString() << std::endl;

#ifdef BUILD_INT4_EXAMPLE
    return ck::utils::check_err(c_m_n_device_result_converted, c_m_n_host_result);
#else
    return ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
#endif
}

bool run_tall_and_skinny_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 5)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
        problem_size.k_batch   = std::stoi(argv[4]);
    }
    else if(argc == 11)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
        problem_size.k_batch   = std::stoi(argv[4]);

        problem_size.M = std::stoi(argv[5]);
        problem_size.N = std::stoi(argv[6]);
        problem_size.K = std::stoi(argv[7]);

        problem_size.stride_A = std::stoi(argv[8]);
        problem_size.stride_B = std::stoi(argv[9]);
        problem_size.stride_C = std::stoi(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4: splitk\n");
        printf("arg5 to 11: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    return run_tall_and_skinny_gemm(problem_size, config);
}
