// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

int run(int argc, char* argv[])
{
    // using ck::host_common::dumpBufferToFile;

    int init_method = 1;

    // GEMM shape for A/B0/B1/C
    // C_g_m_o = A_g_m_k * B0_g_k_n * B1_g_n_o
    ck::index_t M = 1000; // 120
    ck::index_t N = 1000; // 1000
    ck::index_t K = DIM;
    ck::index_t O = DIM;

    // Output shape C[G0, M, G1, O]. Batch dim, outer dim, inner dim must match GEMM shape
    // C_g0_g1_m_o = reshape(C_g_m_o, [g0, g1, m, o])
    // C_g0_m_g1_o = permute(C_g0_g1_m_o, [0, 2, 1, 3])
    ck::index_t G0 = 8;
    ck::index_t G1 = 4;

    bool input_permute  = false;
    bool output_permute = true;

    float p_drop                    = 0.1;
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 2)
    {
        init_method = std::stoi(argv[1]);
    }
    else if(argc == 11)
    {
        init_method = std::stoi(argv[1]);

        M  = std::stoi(argv[2]);
        N  = std::stoi(argv[3]);
        K  = std::stoi(argv[4]);
        O  = std::stoi(argv[5]);
        G0 = std::stoi(argv[6]);
        G1 = std::stoi(argv[7]);

        p_drop = std::stof(argv[8]);

        input_permute  = std::stoi(argv[9]);
        output_permute = std::stoi(argv[10]);
    }
    else
    {
        printf("arg1: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg2 to 7: M, N, K, O, G0, G1\n");
        printf("arg8: drop_prob \n");
        printf("arg9 to 10: input / output permute\n");
        exit(0);
    }

    float alpha = 1.f / std::sqrt(K);

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

    std::vector<ck::index_t> d_gs_ms_ns_lengths{G0, G1, M, N};
    std::vector<ck::index_t> d_gs_ms_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1} // D layout [G0, M, G1, N]
            : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // D layout [G0, G1, M, N]

    std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1} // Z layout [G0, M, G1, N]
            : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // Z layout [G0, G1, M, N]

    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides =
        std::vector<ck::index_t>{G1 * M, M, 1}; // LSE layout [G0, G1, M]

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<Acc0BiasDataType> d_gs_ms_ns(d_gs_ms_ns_lengths, z_gs_ms_ns_strides);
    Tensor<ZDataType> z_gs_ms_ns(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);
    Tensor<LSEDataType> lse_gs_ms_host_result(lse_gs_ms_lengths, lse_gs_ms_strides);
    Tensor<LSEDataType> lse_gs_ms_device_result(lse_gs_ms_lengths, lse_gs_ms_strides);

    Tensor<ZDataType> z_gs_ms_ns_2(z_gs_ms_ns_lengths, z_gs_ms_ns_strides);

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
        d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_2<Acc0BiasDataType>{-1, 1});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_3<Acc0BiasDataType>{-0.5, 0.5});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<Acc0BiasDataType>{1});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<2>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<Acc0BiasDataType>{1});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) *
                           c_gs_ms_os_device_result.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(Acc0BiasDataType) * d_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem z_device_buf(sizeof(ZDataType) * z_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem lse_device_buf(sizeof(LSEDataType) *
                             lse_gs_ms_device_result.mDesc.GetElementSpaceSize());

    DeviceMem z_device_buf_2(sizeof(ZDataType) * z_gs_ms_ns_2.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b0_device_buf.ToDevice(b0_gs_ns_ks.mData.data());
    b1_device_buf.ToDevice(b1_gs_os_ns.mData.data());
    d_device_buf.ToDevice(d_gs_ms_ns.mData.data());

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    auto gemm_op      = DeviceGemmInstance{};
    auto gemm_invoker = gemm_op.MakeInvoker();

    // run for storing z tensor
    auto gemm_arg =
        gemm_op.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                             static_cast<B0DataType*>(b0_device_buf.GetDeviceBuffer()),
                             static_cast<B1DataType*>(b1_device_buf.GetDeviceBuffer()),
                             static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                             static_cast<ZDataType*>(z_device_buf.GetDeviceBuffer()),
                             static_cast<LSEDataType*>(lse_device_buf.GetDeviceBuffer()),
                             static_cast<Acc0BiasDataType*>(d_device_buf.GetDeviceBuffer()),
                             nullptr,
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
                             d_gs_ms_ns_lengths,
                             d_gs_ms_ns_strides,
                             {},
                             {},
                             a_element_op,
                             b0_element_op,
                             acc0_element_op,
                             b1_element_op,
                             c_element_op,
                             p_drop,          // dropout ratio
                             {seed, offset}); // dropout random seed and offset, offset should
                                              // be at least the number of elements on a thread

    if(!gemm_op.IsSupportedArgument(gemm_arg))
    {
        std::cout << gemm_op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    c_device_buf.SetZero();
    lse_device_buf.SetZero();
    gemm_invoker.Run(gemm_arg, StreamConfig{nullptr, false});

    z_device_buf.FromDevice(z_gs_ms_ns.mData.data());

    // dumpBufferToFile("forward_z.dat", z_gs_ms_ns.mData.data(), z_gs_ms_ns.mData.size());

    // do Dropout
    auto dropout_op      = DeviceDropoutInstance();
    auto dropout_invoker = dropout_op.MakeInvoker();

    auto dropout_arg =
        dropout_op.MakeArgument(static_cast<ZDataType*>(z_device_buf_2.GetDeviceBuffer()),
                                z_gs_ms_ns_lengths,
                                z_gs_ms_ns_strides,
                                {seed, offset});

    dropout_invoker.Run(dropout_arg, StreamConfig{nullptr, false});

    z_device_buf_2.FromDevice(z_gs_ms_ns_2.mData.data());

    // dumpBufferToFile("canonic_z.dat", z_gs_ms_ns_2.mData.data(), z_gs_ms_ns_2.mData.size());

    return ck::utils::check_integer_err(z_gs_ms_ns.mData, z_gs_ms_ns_2.mData, 1.0e-5);
}
