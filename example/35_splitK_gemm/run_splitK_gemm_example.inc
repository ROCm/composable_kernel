#pragma once

template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 2e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 2e-1;
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 2e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 2e-1;
    }
    else
    {
        return 1e-3;
    }
}

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t stride_A = K;
    ck::index_t stride_B = K;
    ck::index_t stride_C = N;

    ck::index_t k_batch = 4;
};

struct ExecutionConfig final
{
    // 0 - no verification, 1 - CPU, 2 - GPU, 3 - CPU + GPU
    int do_verification = 1;
    int init_method     = 7;
    bool time_kernel    = false;
};

bool run_splitK_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
    static_assert(sizeof(ADataType) == sizeof(KernelADataType));
    static_assert(sizeof(BDataType) == sizeof(KernelBDataType));
#endif

    auto& [M, N, K, StrideA, StrideB, StrideC, KBatch] = problem_size;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
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
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_device_result.mDesc << std::endl;

    switch(config.init_method)
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
    case 6:
        a_m_k.GenerateTensorValue(GeneratorTensor_PI<ADataType>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    case 7:
        a_m_k.GenerateTensorValue(GeneratorTensor_PI_A<ADataType>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_PI_B<BDataType>{});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
    }

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

#ifdef BUILD_INT4_EXAMPLE
    const Tensor<KernelADataType> a_m_k_converted(a_m_k);
    const Tensor<KernelBDataType> b_k_n_converted(b_k_n);

    a_m_k_device_buf.ToDevice(a_m_k_converted.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_converted.mData.data());
#else
    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
#endif
    c_m_n_device_buf.SetZero();

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(
#ifdef BUILD_INT4_EXAMPLE
        static_cast<KernelADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<KernelBDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
#else
        static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
#endif
        static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideC,
        a_element_op,
        b_element_op,
        c_element_op,
        KBatch);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    invoker.Run(argument, StreamConfig{nullptr, false});
    bool pass = true;

    if((config.do_verification == 1) || (config.do_verification == 3))
    {
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                CElementOp>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        std::cout << "Running verification on CPU." << std::endl;
        ref_invoker.Run(ref_argument);

        if(std::is_same<CDataType, ck::half_t>::value)
        {
            pass &= ck::utils::check_err(
                c_m_n_device_result, c_m_n_host_result, "fp16 incorrect result", 3e-3, 1e-3);
        }
        else
        {
            pass &= ck::utils::check_err(c_m_n_device_result,
                                         c_m_n_host_result,
                                         "Error: Incorrect results!",
                                         get_rtol<CDataType>(),
                                         get_atol<CDataType>());
        }
        if(pass)
            std::cout << "Verification on CPU: PASS" << std::endl;

        if(config.init_method == 6 || config.init_method == 7)
        {
            std::cout << std::fixed << std::setprecision(16);
            AccDataType d = ck::type_convert<AccDataType>(c_m_n_device_result(0, 10));
            AccDataType h = ck::type_convert<AccDataType>(c_m_n_host_result(10, 0));
            std::cout << "device result: " << d << std::endl;
            std::cout << "host result: " << h << std::endl;
            std::cout << "expected result: " << M_PI << std::endl;
            std::cout << "device - host: " << std::abs(d - h) << std::endl;
            std::cout << "device - expected: " << std::abs(d - M_PI) << std::endl;
            std::cout << "atol: " << get_atol<CDataType>() << std::endl;
            std::cout << std::endl << std::endl;
        }
    }

    if((config.do_verification == 2) || (config.do_verification == 3))
    {
        Tensor<CDataType> c_m_n_device_ref_result(
            f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
        DeviceMem c_m_n_device_ref_buf(sizeof(CDataType) *
                                       c_m_n_device_ref_result.mDesc.GetElementSpaceSize());
        // GPU verification
        using ReferenceComputeType = float;
        using ReferenceGemmInstanceGPU =
            ck::tensor_operation::device::ReferenceGemm<ALayout,
                                                        BLayout,
                                                        CLayout,
                                                        ADataType,
                                                        BDataType,
                                                        CDataType,
                                                        AccDataType,
                                                        AElementOp,
                                                        BElementOp,
                                                        CElementOp,
                                                        ReferenceComputeType,
                                                        ReferenceComputeType>;
        auto ref_gemm_gpu    = ReferenceGemmInstanceGPU{};
        auto ref_invoker_gpu = ref_gemm_gpu.MakeInvoker();

        auto ref_argument_gpu = ref_gemm_gpu.MakeArgument(
            static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_m_n_device_ref_buf.GetDeviceBuffer()),
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

        auto gpu_pass = ck::utils::check_err(c_m_n_device_result,
                                             c_m_n_device_ref_result,
                                             "Error: Incorrect results!",
                                             get_rtol<CDataType>(),
                                             get_atol<CDataType>());
        if(gpu_pass)
            std::cout << "Verification on GPU: PASS" << std::endl;

        pass &= gpu_pass;
    }

    if(config.time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 1});

        std::size_t flop = std::size_t(2) * M * N * K;
        std::size_t num_btype =
            sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

        float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
        float gb_per_sec = num_btype / 1.E6 / ave_time;
        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }

    return pass;
}

bool run_splitK_gemm_example(int argc, char* argv[])
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
        printf("arg1: verification (0=no, 1=CPU, 2=GPU, 3=CPU and GPU)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4: KBatch\n");
        printf("arg5 to 11: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    return run_splitK_gemm(problem_size, config);
}
