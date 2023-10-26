// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;

    // GEMM shape for A/B0/B1/C
    // C_g_m_o = A_g_m_k * B0_g_k_n * B1_g_n_o
    ck::index_t M = 1000; // 120
    ck::index_t N = 1000; // 1000
    ck::index_t K = DIM;
    ck::index_t O = DIM;

    // Output shape C[G0, M, G1Q, O]. Batch dim, outer dim, inner dim must match GEMM shape
    // C_g0_g1q_m_o = reshape(C_g_m_o, [g0, g1q, m, o])
    // C_g0_m_g1q_o = permute(C_g0_g1q_m_o, [0, 2, 1, 3])
    ck::index_t G0   = 7;
    ck::index_t G1Q  = 12; // h_q
    ck::index_t G1KV = 12; // h_kv

    bool input_permute  = false;
    bool output_permute = true;

    float p_drop                    = 0.1;
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 14)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M    = std::stoi(argv[4]);
        N    = std::stoi(argv[5]);
        K    = std::stoi(argv[6]);
        O    = std::stoi(argv[7]);
        G0   = std::stoi(argv[8]);
        G1Q  = std::stoi(argv[9]);
        G1KV = std::stoi(argv[10]);

        p_drop = std::stof(argv[11]);

        input_permute  = std::stoi(argv[12]);
        output_permute = std::stoi(argv[13]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 10: M, N, K, O, G0, G1Q, G1KV\n");
        printf("arg11: p_drop\n");
        printf("arg12 to 13: input / output permute\n");
        exit(0);
    }

    float p_dropout                = 1 - p_drop;
    ZDataType p_dropout_in_uint8_t = ZDataType(std::floor(p_dropout * 255.0));
    float rp_dropout               = 1.0 / p_dropout;
    float alpha                    = 1.f / std::sqrt(K);

    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1Q, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1Q * K, K, G1Q * K, 1} // A layout [G0, M, G1Q, K]
            : std::vector<ck::index_t>{G1Q * M * K, M * K, K, 1};  // A layout [G0, G1Q, M, K]

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1KV, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1KV * K, K, G1KV * K, 1} // B0 layout [G0, N, G1KV, K]
            : std::vector<ck::index_t>{G1KV * N * K, N * K, K, 1};   // B0 layout [G0, G1KV, N, K]

    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1KV, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1KV * O, O, 1, G1KV * O} // B1 layout [G0, N, G1KV, O]
            : std::vector<ck::index_t>{G1KV * N * O, N * O, 1, O};   // B1 layout [G0, G1KV, N, O]

    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1Q, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1Q * O, O, G1Q * O, 1} // C layout [G0, M, G1Q, O]
            : std::vector<ck::index_t>{G1Q * M * O, M * O, O, 1};  // C layout [G0, G1Q, M, O]

    std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1Q, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1Q * N, N, G1Q * N, 1} // Z layout [G0, M, G1Q, N]
            : std::vector<ck::index_t>{G1Q * M * N, M * N, N, 1};  // Z layout [G0, G1Q, M, N]

    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1Q, M};
    std::vector<ck::index_t> lse_gs_ms_strides =
        std::vector<ck::index_t>{G1Q * M, M, 1}; // LSE layout [G0, G1Q, M]

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<ZDataType> z_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
    Tensor<LSEDataType> lse_gs_ms_host_result(lse_gs_ms_lengths, lse_gs_ms_strides);
    Tensor<LSEDataType> lse_gs_ms_device_result(lse_gs_ms_lengths, lse_gs_ms_strides);

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b0_gs_ns_ks: " << b0_gs_ns_ks.mDesc << std::endl;
    std::cout << "b1_gs_os_ns: " << b1_gs_os_ns.mDesc << std::endl;
    std::cout << "c_gs_ms_os: " << c_gs_ms_os_host_result.mDesc << std::endl;
    std::cout << "z_gs_ms_ns: " << z_gs_ms_ns.mDesc << std::endl;
    std::cout << "lse_gs_ms_os: " << lse_gs_ms_host_result.mDesc << std::endl;

    z_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<ZDataType>{0});

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<2>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) *
                           c_gs_ms_os_device_result.mDesc.GetElementSpaceSize());
    DeviceMem z_device_buf(sizeof(ZDataType) * z_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem lse_device_buf(sizeof(LSEDataType) *
                             lse_gs_ms_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b0_device_buf.ToDevice(b0_gs_ns_ks.mData.data());
    b1_device_buf.ToDevice(b1_gs_os_ns.mData.data());
    z_device_buf.ToDevice(z_gs_ms_ns.mData.data());

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    // TODO ANT: replace array with vector?
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(
        static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
        static_cast<B0DataType*>(b0_device_buf.GetDeviceBuffer()),
        static_cast<B1DataType*>(b1_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
        static_cast<ZDataType*>(nullptr),
        static_cast<LSEDataType*>(lse_device_buf.GetDeviceBuffer()),
        nullptr, // std::array<void*, 1> p_acc0_biases;
        nullptr, // std::array<void*, 1> p_acc1_biases;
        a_gs_ms_ks_lengths,
        a_gs_ms_ks_strides,
        b0_gs_ns_ks_lengths,
        b0_gs_ns_ks_strides,
        b1_gs_os_ns_lengths,
        b1_gs_os_ns_strides,
        c_gs_ms_os_lengths,
        c_gs_ms_os_strides,
        z_gs_ms_ns_lengths,
        z_gs_ms_ns_strides,
        lse_gs_ms_lengths,
        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
        a_element_op,
        b0_element_op,
        acc0_element_op,
        b1_element_op,
        c_element_op,
        p_drop,          // dropout ratio
        {seed, offset}); // dropout random seed and offset, offset should be at least the number of
                         // elements on a thread

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    ck::index_t BatchCount = G0 * G1Q;

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
    std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                             sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                            BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(do_verification)
    {

        // run for storing z tensor
        argument = gemm.MakeArgument(
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<B0DataType*>(b0_device_buf.GetDeviceBuffer()),
            static_cast<B1DataType*>(b1_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
            static_cast<ZDataType*>(z_device_buf.GetDeviceBuffer()),
            static_cast<LSEDataType*>(lse_device_buf.GetDeviceBuffer()),
            {}, // std::array<void*, 1> p_acc0_biases;
            {}, // std::array<void*, 1> p_acc1_biases;
            a_gs_ms_ks_lengths,
            a_gs_ms_ks_strides,
            b0_gs_ns_ks_lengths,
            b0_gs_ns_ks_strides,
            b1_gs_os_ns_lengths,
            b1_gs_os_ns_strides,
            c_gs_ms_os_lengths,
            c_gs_ms_os_strides,
            z_gs_ms_ns_lengths,
            z_gs_ms_ns_strides,
            lse_gs_ms_lengths,
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
            a_element_op,
            b0_element_op,
            acc0_element_op,
            b1_element_op,
            c_element_op,
            p_drop,          // dropout ratio
            {seed, offset}); // dropout random seed and offset, offset should be at least the number
                             // of elements on a thread
        c_device_buf.SetZero();
        lse_device_buf.SetZero();
        invoker.Run(argument, StreamConfig{nullptr, false});

        c_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());
        z_device_buf.FromDevice(z_gs_ms_ns.mData.data());
        lse_device_buf.FromDevice(lse_gs_ms_device_result.mData.data());

        Tensor<ADataType> a_g_m_k({BatchCount, M, K});
        Tensor<B0DataType> b0_g_k_n({BatchCount, K, N});
        Tensor<B1DataType> b1_g_n_o({BatchCount, N, O});
        Tensor<AccDataType> acc0_g_m_n({BatchCount, M, N}); // scratch object after gemm0
        Tensor<ADataType> a1_g_m_n({BatchCount, M, N});     // scratch object after softmax
        Tensor<ADataType> a1_g_m_n_drop({BatchCount, M, N});
        Tensor<LSEDataType> lse_g_m_host_result(
            {BatchCount, M}); // scratch object after max + ln(sum)
        Tensor<ZDataType> z_g_m_n({BatchCount, M, N});
        Tensor<CDataType> c_g_m_o_host_result({BatchCount, M, O}); // scratch object after gemm1

        // permute
        a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
            a_g_m_k(idx[0] * G1Q + idx[1], idx[2], idx[3]) = self(idx);
        });
        b0_g_k_n.ForEach([&](auto& self, auto idx) {
            const size_t& g0   = idx[0] / G1Q;
            const size_t& g1q  = idx[0] % G1Q;
            const size_t& g1kv = g1q / (G1Q / G1KV);

            self(idx) = b0_gs_ns_ks(g0, g1kv, idx[2], idx[1]);
        });
        b1_g_n_o.ForEach([&](auto& self, auto idx) {
            const size_t& g0   = idx[0] / G1Q;
            const size_t& g1q  = idx[0] % G1Q;
            const size_t& g1kv = g1q / (G1Q / G1KV);

            self(idx) = b1_gs_os_ns(g0, g1kv, idx[2], idx[1]);
        });
        z_gs_ms_ns.ForEach([&](auto& self, auto idx) {
            z_g_m_n(idx[0] * G1Q + idx[1], idx[2], idx[3]) = self(idx);
        });

        // gemm 0
        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        // masking
        const auto mask = DeviceGemmInstance::C0MatrixMask(M, N);
        acc0_g_m_n.ForEach([&](auto& self, auto idx) {
            if(mask.IsMaskedElement(idx[1], idx[2]))
                self(idx) = -ck::NumericLimits<float>::Infinity();
        });

        // softmax
        auto ref_softmax         = ReferenceSoftmaxInstance{};
        auto ref_softmax_invoker = ref_softmax.MakeInvoker();
        auto ref_softmax_argument =
            ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2}, &lse_g_m_host_result);

        ref_softmax_invoker.Run(ref_softmax_argument);

        // dropout after softmax
        auto ref_dropout         = ReferenceDropoutInstance{};
        auto ref_dropout_invoker = ref_dropout.MakeInvoker();
        auto ref_dropout_argment = ref_dropout.MakeArgument(
            z_g_m_n, a1_g_m_n, a1_g_m_n_drop, p_dropout_in_uint8_t, rp_dropout);
        ref_dropout_invoker.Run(ref_dropout_argment);

        // gemm1
        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(a1_g_m_n_drop,
                                                         b1_g_n_o,
                                                         c_g_m_o_host_result,
                                                         PassThrough{},
                                                         b1_element_op,
                                                         c_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // permute
        c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0  = idx[0];
            const size_t& g1q = idx[1];

            const size_t g = g0 * G1Q + g1q;

            self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
        });
        lse_gs_ms_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0  = idx[0];
            const size_t& g1q = idx[1];

            const size_t g = g0 * G1Q + g1q;

            self(idx) = lse_g_m_host_result(g, idx[2]);
        });

        // default absolute error and relative error is 0.001
        double rtol = 1e-3;
        double atol = 1e-3;

        // when BF16 is taken, set absolute error and relative error to 0.01
        if(std::is_same_v<DataType, ck::bhalf_t> || std::is_same_v<GemmDataType, ck::bhalf_t>)
        {
            rtol = 1e-2;
            atol = 1e-2;
        }

        return ck::utils::check_err(c_gs_ms_os_device_result.mData,
                                    c_gs_ms_os_host_result.mData,
                                    "Error: Incorrect results c!",
                                    rtol,
                                    atol) &&
                       ck::utils::check_err(lse_gs_ms_device_result.mData,
                                            lse_gs_ms_host_result.mData,
                                            "Error: Incorrect results lse!",
                                            rtol,
                                            atol)
                   ? 0
                   : 1;
    }

    return 0;
}
