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
    ck_tile::index_t scale_n_size = N / 32;
    ck_tile::index_t scale_k_size = K / 32;
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_a_host(
        ck_tile::HostTensorDescriptor({M, scale_k_size}, {scale_k_size, 1}));
    ck_tile::HostTensor<ck_tile::e8m0_t> scale_b_host(
        ck_tile::HostTensorDescriptor({scale_k_size, scale_n_size}, {scale_n_size, 1}));
    switch(init_method)
    {
    case 0:
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_host);
        ck_tile::FillUniformDistribution<ck_tile::e8m0_t>{-1.f, 1.f}(scale_a_host);
        ck_tile::FillUniformDistribution<ck_tile::e8m0_t>{-1.f, 1.f}(scale_b_host);
        break;
    case 1:
        ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_host);
        ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_host);
        ck_tile::FillConstant<ck_tile::e8m0_t>{ck_tile::e8m0_t(1.f)}(scale_a_host);
        ck_tile::FillConstant<ck_tile::e8m0_t>{ck_tile::e8m0_t(1.f)}(scale_b_host);
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
    using ScaleM = ck_tile::MXScalePointer<1,  32>; // per-token
    using ScaleN = ck_tile::MXScalePointer<32, 32>; // per-block

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

    if(validation > 0)
    {
        c_dev_buf.FromDevice(c_host.data());
        // TODO: Implement validation logic (reference GEMM with scales)
        // For now just print success if it runs
        std::cout << "Validation not implemented yet." << std::endl;
    }
    return 0;
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
    int persistent_opt   = arg_parser.get_int("persistent");

    if(a_layout == "R" && b_layout == "C")
    {
        if(mx_prec == "fp4" || mx_prec == "fp4xfp4")
        {
            if(persistent_opt == 0)
                return run_mx_gemm_with_layouts<ck_tile::pk_fp4_t,
                                                  ck_tile::pk_fp4_t,
                                                  float,
                                                  MXfp4_GemmConfig16,
                                                  false>(argc, argv, Row{}, Col{}, Row{});
            else
                throw std::runtime_error("Only non-persistent kernels are supported currently!");
        }
        else
        {
             throw std::runtime_error("Only fp4xfp4 is supported currently!");
        }
    }
    else
    {
        throw std::runtime_error("Only A=Row, B=Col layout is supported currently!");
    }
    return 0;
}
