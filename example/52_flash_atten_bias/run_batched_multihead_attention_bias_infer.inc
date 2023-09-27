// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape for A/B0/B1/C
    // C_g_m_o = A_g_m_k * B0_g_k_n * B1_g_n_o
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = DIM;
    ck::index_t O = DIM;

    // Output shape C[G0, M, G1, O]. Batch dim, outer dim, inner dim must match GEMM shape
    // C_g0_g1_m_o = reshape(C_g_m_o, [g0, g1, m, o])
    // C_g0_m_g1_o = permute(C_g0_g1_m_o, [0, 2, 1, 3])
    ck::index_t G0 = 7;
    ck::index_t G1 = 13;

    float alpha = 1;

    bool input_permute  = false;
    bool output_permute = true;

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
    else if(argc == 13)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M  = std::stoi(argv[4]);
        N  = std::stoi(argv[5]);
        K  = std::stoi(argv[6]);
        O  = std::stoi(argv[7]);
        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);

        alpha = std::stof(argv[10]);

        input_permute  = std::stoi(argv[11]);
        output_permute = std::stoi(argv[12]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        printf("arg11 to 12: input / output permute\n");
        exit(0);
    }

    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // A layout [G0, M, G1, K]
            : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // A layout [G0, G1, M, K]

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // B0 layout [G0, N, G1, K]
            : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // B0 layout [G0, G1, N, K]

    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // B1 layout [G0, N, G1, O]
            : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // B1 layout [G0, G1, N, O]

    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // C layout [G0, M, G1, O]
            : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // C layout [G0, G1, M, O]

    std::vector<ck::index_t> d0_gs_ms_ns_lengths{G0, G1, M, N};
    std::vector<ck::index_t> d0_gs_ms_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1} // D0 layout [G0, M, G1, N]
            : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // D0 layout [G0, G1, M, N]

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    Tensor<Acc0BiasDataType> d0_gs_ms_ns(d0_gs_ms_ns_lengths, d0_gs_ms_ns_strides);
    Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b0_gs_ns_ks: " << b0_gs_ns_ks.mDesc << std::endl;
    std::cout << "b1_gs_os_ns: " << b1_gs_os_ns.mDesc << std::endl;
    std::cout << "c_gs_ms_os: " << c_gs_ms_os_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        d0_gs_ms_ns.GenerateTensorValue(GeneratorTensor_2<Acc0BiasDataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        d0_gs_ms_ns.GenerateTensorValue(GeneratorTensor_3<Acc0BiasDataType>{-0.5, 0.5});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        d0_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<Acc0BiasDataType>{1});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<2>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        d0_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<Acc0BiasDataType>{1});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(Acc0BiasDataType) * d0_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) *
                           c_gs_ms_os_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b0_device_buf.ToDevice(b0_gs_ns_ks.mData.data());
    d0_device_buf.ToDevice(d0_gs_ms_ns.mData.data());
    b1_device_buf.ToDevice(b1_gs_os_ns.mData.data());

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
        static_cast<Acc0BiasDataType*>(d0_device_buf.GetDeviceBuffer()), // p_acc0_bias;
        nullptr,                                                         // p_acc1_bias;
        a_gs_ms_ks_lengths,
        a_gs_ms_ks_strides,
        b0_gs_ns_ks_lengths,
        b0_gs_ns_ks_strides,
        b1_gs_os_ns_lengths,
        b1_gs_os_ns_strides,
        c_gs_ms_os_lengths,
        c_gs_ms_os_strides,
        d0_gs_ms_ns_lengths, // acc0_bias_gs_ms_ns_lengths
        d0_gs_ms_ns_strides, // acc0_bias_gs_ms_ns_strides
        {},                  // std::vector<ck::index_t>{acc1_biases_gs_ms_os_lengths},
        {},                  // std::vector<ck::index_t>{acc1_biases_gs_ms_os_strides},
        a_element_op,
        b0_element_op,
        acc0_element_op,
        b1_element_op,
        c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    ck::index_t BatchCount = G0 * G1;

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2 + size_t(M) * N) * BatchCount;
    std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                             sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O + sizeof(Acc0BiasDataType) * M * N) *
                            BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(do_verification)
    {
        c_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

        Tensor<ADataType> a_g_m_k({BatchCount, M, K});
        Tensor<B0DataType> b0_g_k_n({BatchCount, K, N});
        Tensor<B1DataType> b1_g_n_o({BatchCount, N, O});
        Tensor<AccDataType> acc0_g_m_n({BatchCount, M, N});        // scratch object after gemm0
        Tensor<Acc0BiasDataType> d0_g_m_n({BatchCount, M, N});
        Tensor<ADataType> a1_g_m_n({BatchCount, M, N});            // scratch object after softmax
        Tensor<CDataType> c_g_m_o_host_result({BatchCount, M, O}); // scratch object after gemm1

        // permute
        a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
            a_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        b0_gs_ns_ks.ForEach([&](auto& self, auto idx) {
            b0_g_k_n(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });
        d0_gs_ms_ns.ForEach([&](auto& self, auto idx) {
            d0_g_m_n(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        b1_gs_os_ns.ForEach([&](auto& self, auto idx) {
            b1_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });

        // gemm 0
        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        // bias
        acc0_g_m_n.ForEach([&](auto& self, auto idx) {
            self(idx) += ck::type_convert<AccDataType>(d0_g_m_n(idx));
        });

        // masking
        const auto mask = DeviceGemmInstance::C0MatrixMask(M, N);
        acc0_g_m_n.ForEach([&](auto& self, auto idx) {
            if(mask.IsMaskedElement(idx[1], idx[2]))
                self(idx) = -ck::NumericLimits<float>::Infinity();
        });

        // softmax
        auto ref_softmax          = ReferenceSoftmaxInstance{};
        auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
        auto ref_softmax_argument = ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2});

        ref_softmax_invoker.Run(ref_softmax_argument);

        // gemm1
        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            a1_g_m_n, b1_g_n_o, c_g_m_o_host_result, PassThrough{}, b1_element_op, c_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // permute
        c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
        });

        // default absolute error and relative error is 0.001
        double rtol = 1e-3;
        double atol = 1e-3;

        // when BF16 is taken, set absolute error and relative error to 0.01
        if(std::is_same_v<ADataType, ck::bhalf_t> && std::is_same_v<B0DataType, ck::bhalf_t> &&
           std::is_same_v<B1DataType, ck::bhalf_t> && std::is_same_v<CDataType, ck::bhalf_t>)
        {
            rtol = 1e-2;
            atol = 1e-2;
        }

        return ck::utils::check_err(c_gs_ms_os_device_result.mData,
                                    c_gs_ms_os_host_result.mData,
                                    "Error: Incorrect results!",
                                    rtol,
                                    atol)
                   ? 0
                   : 1;
    }

    return 0;
}
