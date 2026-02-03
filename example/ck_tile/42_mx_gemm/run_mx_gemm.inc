// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename GemmConfig,
          bool UsePersistentKernel,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_gemm_with_layouts(int argc,
                             char* argv[],
                             ALayout,
                             BLayout,
                             CLayout)
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    ck_tile::index_t M        = arg_parser.get_int("m");
    ck_tile::index_t N        = arg_parser.get_int("n");
    ck_tile::index_t K        = arg_parser.get_int("k");
    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");
    int validation            = arg_parser.get_int("v");
    int n_warmup              = arg_parser.get_int("warmup");
    int n_repeat              = arg_parser.get_int("repeat");
    int kbatch                = arg_parser.get_int("split_k");
    int init_method           = arg_parser.get_int("init");

    using CDataType = ck_tile::fp16_t;

    if(stride_A == 0)
        stride_A = is_row_major(ALayout{}) ? K : M;
    if(stride_B == 0)
        stride_B = is_row_major(BLayout{}) ? N : K;
    if(stride_C == 0)
        stride_C = is_row_major(CLayout{}) ? N : M;

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::HostTensorDescriptor({M, K}, {stride_A, 1}));
    ck_tile::HostTensor<BDataType> b_host(
        ck_tile::HostTensorDescriptor({K, N}, {stride_B, 1}));
    ck_tile::HostTensor<CDataType> c_host(
        ck_tile::HostTensorDescriptor({M, N}, {stride_C, 1}));

    // Scale tensors
    // Assuming block scale 32
    using ScaleType = ck_tile::e8m0_t;
    ck_tile::index_t scale_k_size = K / 32;
    ck_tile::HostTensor<ScaleType> scale_a_host(
        ck_tile::HostTensorDescriptor({M, scale_k_size}, {scale_k_size, 1}));
    ck_tile::HostTensor<ScaleType> scale_b_host(
        ck_tile::HostTensorDescriptor({scale_k_size, N}, {1, scale_k_size}));
    switch(init_method)
    {
    case 0:
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_host);
        ck_tile::FillUniformDistribution<ScaleType>{1.f, 10.f}(scale_a_host);
        ck_tile::FillUniformDistribution<ScaleType>{1.f, 10.f}(scale_b_host);
        break;
    case 1:
        ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_host);
        ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_host);
        break;
    case 2:
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_host);
        break;
    }

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_a_dev_buf(scale_a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_host.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    b_dev_buf.ToDevice(b_host.data());
    scale_a_dev_buf.ToDevice(scale_a_host.data());
    scale_b_dev_buf.ToDevice(scale_b_host.data());

    // Scale pointers
    using ScaleM = ck_tile::MXScalePointer<1, 32>; // in blocks of 32 in K
    using ScaleN = ck_tile::MXScalePointer<1, 32>;

    ScaleM scale_m(reinterpret_cast<float*>(scale_a_dev_buf.GetDeviceBuffer()));
    ScaleN scale_n(reinterpret_cast<float*>(scale_b_dev_buf.GetDeviceBuffer()));

    float ave_time = invoke_mx_gemm<GemmConfig,
                                    ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout,
                                    ScaleM,
                                    ScaleN,
                                    UsePersistentKernel>(
        a_dev_buf, b_dev_buf, c_dev_buf, M, N, K, stride_A, stride_B, stride_C, kbatch, scale_m, scale_n, n_warmup, n_repeat);
    
    (void)ave_time;

    bool pass = true;
    if(validation > 0)
    {
        // get output data from device
        c_dev_buf.FromDevice(c_host.data());

        // compute reference
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
            a_host, b_host, c_m_n_host_ref, scale_a_host, scale_b_host);
        // ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
        //     a_host, b_host, c_m_n_host_ref);

        auto calculate_rtol_atol = [&K, &kbatch](const float max_accumulated_value)
        {
            using ComputeType =
                std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
            // Calculate thresholds
            const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
                ck_tile::integer_divide_ceil(K, kbatch));
            const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
                max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
            // Calculate error due to split_k accumulation
            const auto rtol_split_k =
                ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
            const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
                max_accumulated_value, kbatch);
            // Use higher threshold
            return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
        };

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto [rtol, atol] = calculate_rtol_atol(max_accumulated_value);

        pass = ck_tile::check_err(
            c_host, c_m_n_host_ref, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }
    return pass ? 0 : -1;
}

int run_mx_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string mx_prec  = arg_parser.get_str("mx_prec");
    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "C")
    {
        if(mx_prec == "fp4" || mx_prec == "fp4xfp4")
        {
            return run_mx_gemm_with_layouts<ck_tile::pk_fp4_t,
                                                ck_tile::pk_fp4_t,
                                                float,
                                                MXfp4_GemmConfig16,
                                                true>(argc, argv, Row{}, Col{}, Row{});
        }
        else if(mx_prec == "fp8" || mx_prec == "fp8xfp8")
        {
            return run_mx_gemm_with_layouts<ck_tile::fp8_t,
                                                ck_tile::fp8_t,
                                                float,
                                                MXfp8_GemmConfig16,
                                                true>(argc, argv, Row{}, Col{}, Row{});
        }
        else
        {
             throw std::runtime_error("Only fp4/8 is supported currently!");
        }
    }
    else
    {
        throw std::runtime_error("Only A=Row, B=Col layout is supported currently!");
    }
    return 0;
}
