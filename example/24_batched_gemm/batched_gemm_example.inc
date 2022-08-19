#pragma once

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t stride_A = K;
    ck::index_t stride_B = K;
    ck::index_t stride_C = N;

    ck::index_t batch_stride_A = M * K;
    ck::index_t batch_stride_B = K * N;
    ck::index_t batch_stride_C = M * N;

    ck::index_t batch_count = 16;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

bool run_batched_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto& [M, N, K, stride_A, stride_B, stride_C, batch_stride_A, batch_stride_B, batch_stride_C, batch_count] = problem_size;

    // GEMM shape
    auto f_host_tensor_descriptor = [](std::size_t batch_count_,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count_, row, col}),
                                        std::vector<std::size_t>({batch_stride, stride, 1}));
        }
        else
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count_, row, col}),
                                        std::vector<std::size_t>({batch_stride, 1, stride}));
        }
    };

    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(batch_count, M, K, stride_A, batch_stride_A, ALayout{}));
    Tensor<BDataType> b_g_k_n(
        f_host_tensor_descriptor(batch_count, K, N, stride_B, batch_stride_B, BLayout{}));

    Tensor<EDataType> e_g_m_n_device_result(
        f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, ELayout{}));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "e_g_m_n: " << e_g_m_n_device_result.mDesc << std::endl;

    switch(config.init_method)
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

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(EDataType) * e_g_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    // do GEMM
    auto argument = gemm.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                      b_device_buf.GetDeviceBuffer(),
                                      {},
                                      c_device_buf.GetDeviceBuffer(),
                                      M,
                                      N,
                                      K,
                                      batch_count,
                                      stride_A,
                                      stride_B,
                                      {},
                                      stride_C,
                                      batch_stride_A,
                                      batch_stride_B,
                                      {},
                                      batch_stride_C,
                                      a_element_op,
                                      b_element_op,
                                      cde_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop      = std::size_t(2) * batch_count * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * batch_count * M * K +
                            sizeof(BDataType) * batch_count * K * N +
                            sizeof(EDataType) * batch_count * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;

    if(config.do_verification)
    {
        c_device_buf.FromDevice(e_g_m_n_device_result.mData.data());

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        Tensor<EDataType> e_g_m_n_host_result(
            f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, ELayout{}));

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, e_g_m_n_host_result, a_element_op, b_element_op, cde_element_op);

        ref_invoker.Run(ref_argument);

        pass = ck::utils::check_err(
            e_g_m_n_host_result.mData, e_g_m_n_device_result.mData, "Error: Incorrect results c");
    }

    return pass ? 0 : 1;
}

bool run_batched_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    problem_size.M = 256 * (rand() % 16 + 1);
    problem_size.N = 128 * (rand() % 16 + 1);
    problem_size.K = 64 * (rand() % 16 + 1);

    problem_size.stride_A = problem_size.K;
    problem_size.stride_B = problem_size.K;
    problem_size.stride_C = problem_size.N;

    problem_size.batch_stride_A = problem_size.M * problem_size.K;
    problem_size.batch_stride_B = problem_size.K * problem_size.N;
    problem_size.batch_stride_C = problem_size.M * problem_size.N;

    problem_size.batch_count = 16;

    if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        exit(0);
    }

    return run_batched_gemm(problem_size, config);
}
