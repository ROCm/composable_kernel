/ Add after the existing content

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,  
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename GemmFunc>
int run_gemm_example_with_layouts(int argc, char* argv[],
                                 ALayout a_layout,
                                 BLayout b_layout,
                                 CLayout c_layout,
                                 GemmFunc gemm_func)
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    // Parse Stream-K specific arguments if present
    float grid_multiplier = arg_parser.exists("grid_multiplier") ? 
                           arg_parser.get_float("grid_multiplier") : 1.0f;
    int streamk_mode = arg_parser.exists("streamk_mode") ? 
                      arg_parser.get_int("streamk_mode") : 0;
    int sk_blocks = arg_parser.exists("sk_blocks") ? 
                   arg_parser.get_int("sk_blocks") : 0;

    const int M = arg_parser.get_int("m");
    const int N = arg_parser.get_int("n");
    const int K = arg_parser.get_int("k");
    const int k_batch = arg_parser.get_int("k_batch");

    const int seed = arg_parser.get_int("seed");
    const int do_validation = arg_parser.get_int("validate");
    const int do_berr = arg_parser.get_int("berr");
    const int time_kernel = arg_parser.get_int("time_kernel");
    const int verbose = arg_parser.get_int("verbose");

    // Calculate strides
    const int stride_a = is_row_major(a_layout) ? K : M;
    const int stride_b = is_row_major(b_layout) ? N : K;
    const int stride_c = is_row_major(c_layout) ? N : M;

    // Allocate memory
    auto f_host_a = std::make_unique<HostTensorFunctor<ADataType>>(M, K, stride_a, seed);
    auto f_host_b = std::make_unique<HostTensorFunctor<BDataType>>(K, N, stride_b, seed);
    auto f_host_c = std::make_unique<HostTensorFunctor<CDataType>>(M, N, stride_c, 0);

    if(do_validation || do_berr)
    {
        // Perform reference GEMM
        if constexpr(is_same_v<AccDataType, int32_t>)
        {
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                f_host_a->data, f_host_b->data, f_host_c->data,
                M, N, K, stride_a, stride_b, stride_c,
                a_layout, b_layout, c_layout);
        }
        else
        {
            ck_tile::reference_gemm<ADataType, BDataType, float, CDataType>(
                f_host_a->data, f_host_b->data, f_host_c->data,
                M, N, K, stride_a, stride_b, stride_c,
                a_layout, b_layout, c_layout);
        }
    }

    auto f_device_a = f_host_a->CreateDeviceTensor();
    auto f_device_b = f_host_b->CreateDeviceTensor();
    auto f_device_c_result = std::make_unique<DeviceTensorFunctor<CDataType>>(M, N, stride_c);

    // Create arguments based on whether this is Stream-K or regular GEMM
    if (streamk_mode > 0)
    {
        ck_tile::StreamKGemmHostArgs<> args{
            f_device_a->data, f_device_b->data, {}, f_device_c_result->data,
            k_batch, M, N, K, stride_a, stride_b, {}, stride_c,
            streamk_mode, 0, grid_multiplier, sk_blocks, 0};
            
        hipGetErrorString(hipMemset(args.e_ptr, 0, M * N * sizeof(CDataType)));
        hipDeviceSynchronize();

        const float ave_time = gemm_func(args, 
            ck_tile::stream_config{time_kernel, 0, verbose, false, 20});

        if(ave_time < 0)
            return 1;

        const double gflops = 2.0 * M * N * K / 1e9;
        const double tflops = gflops / ave_time;

        std::cout << "Stream-K Perf: " << ave_time << " ms, " << tflops << " TFlops, " 
                  << gflops << " GFlop" << std::endl;
    }
    else
    {
        ck_tile::GemmHostArgs<> args{
            f_device_a->data, f_device_b->data, {}, f_device_c_result->data,
            k_batch, M, N, K, stride_a, stride_b, {}, stride_c};
            
        hipGetErrorString(hipMemset(args.e_ptr, 0, M * N * sizeof(CDataType)));
        hipDeviceSynchronize();

        const float ave_time = gemm_func(args,
            ck_tile::stream_config{time_kernel, 0, verbose, false, 20});

        if(ave_time < 0)
            return 1;

        const double gflops = 2.0 * M * N * K / 1e9;
        const double tflops = gflops / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " 
                  << gflops << " GFlop" << std::endl;
    }

    if(do_validation || do_berr)
    {
        auto f_host_c_result = f_device_c_result->CreateHostTensor();
        
        auto [rtol, atol] = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, k_batch, std::numeric_limits<AccDataType>::max());
            
        auto [is_valid, num_errors, max_diff, max_relative_diff] = 
            ck_tile::check_err(f_host_c->data, f_host_c_result->data, M * N,
                             ck_tile::error_tolerance{rtol, atol});
                             
        if(verbose)
        {
            std::cout << "max_diff: " << max_diff 
                      << ", max_relative_diff: " << max_relative_diff
                      << ", rtol: " << rtol << ", atol: " << atol << std::endl;
        }

        return !is_valid;
    }

    return 0;
}