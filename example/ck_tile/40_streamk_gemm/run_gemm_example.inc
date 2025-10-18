// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

template <typename Layout>
static constexpr inline auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to multiple WGs working in the same C macro tile
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough,
          ck_tile::StreamKReductionStrategy ReductionStrategy>
std::tuple<float, ck_tile::index_t> gemm(const ck_tile::StreamKHostArgs& args,
                                         const ck_tile::stream_config& s);

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
std::tuple<float, ck_tile::index_t>
invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
            ck_tile::DeviceMem& b_k_n_dev_buf,
            ck_tile::DeviceMem& c_m_n_dev_buf,
            ck_tile::index_t M,
            ck_tile::index_t N,
            ck_tile::index_t K,
            ck_tile::index_t stride_A,
            ck_tile::index_t stride_B,
            ck_tile::index_t stride_C,
            int n_warmup,
            int n_repeat,
            bool flush_cache,
            ck_tile::StreamKReductionStrategy reduction_strategy,
            uint32_t num_sk_blocks)
{
    ck_tile::StreamKHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
                                  b_k_n_dev_buf.GetDeviceBuffer(),
                                  c_m_n_dev_buf.GetDeviceBuffer(),
                                  M,
                                  N,
                                  K,
                                  stride_A,
                                  stride_B,
                                  stride_C,
                                  reduction_strategy,
                                  num_sk_blocks};

    std::tuple<float, ck_tile::index_t> ave_time_and_batch;

    if(args.reduction_strategy == ck_tile::StreamKReductionStrategy::Atomic)
    {
        ave_time_and_batch = gemm<GemmConfig,
                                  ADataType,
                                  BDataType,
                                  DsDataType,
                                  AccDataType,
                                  CDataType,
                                  ALayout,
                                  BLayout,
                                  DsLayout,
                                  CLayout,
                                  CDEElementWise,
                                  ck_tile::StreamKReductionStrategy::Atomic>(
            args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, flush_cache});
    }
    else /*Reduction*/
    {
        ave_time_and_batch = gemm<GemmConfig,
                                  ADataType,
                                  BDataType,
                                  DsDataType,
                                  AccDataType,
                                  CDataType,
                                  ALayout,
                                  BLayout,
                                  DsLayout,
                                  CLayout,
                                  CDEElementWise,
                                  ck_tile::StreamKReductionStrategy::Reduction>(
            args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, flush_cache});
    }

    return ave_time_and_batch;
}

template <typename CDataType>
bool do_verify(const ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
               const ck_tile::HostTensor<CDataType>& c_m_n_ref,
               const ck_tile::tuple<double, double>& rtol_atol,
               const char* variant)
{
    bool pass = ck_tile::check_err(c_m_n_dev_result,
                                   c_m_n_ref,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
              << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The " << variant << " verification result is:" << (pass ? "correct" : "fail")
              << std::endl;
    return pass;
}

ck_tile::StreamKReductionStrategy get_reduction_strategy_value(const std::string& strategy)
{
    if(strategy == "atomic")
    {
        return ck_tile::StreamKReductionStrategy::Atomic;
    }
    else if(strategy == "reduction")
    {
        return ck_tile::StreamKReductionStrategy::Reduction;
    }
    else
    {
        throw std::runtime_error("Unsupported Stream-K reduction strategy !!!");
    }
}

template <typename GemmConfig,
          typename TypeConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_gemm_example_with_layouts(int argc,
                                  char* argv[],
                                  const ALayout a_layout                  = ALayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    static_assert(!GemmConfig::Preshuffle, "Not implemented");
    static_assert(!GemmConfig::UseStructuredSparsity, "Not implemented");
    static_assert(!GemmConfig::PermuteA, "Not implemented");
    static_assert(!GemmConfig::PermuteB, "Not implemented");

    using ADataType   = typename TypeConfig::ADataType;
    using BDataType   = typename TypeConfig::BDataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using CDataType   = typename TypeConfig::CDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    bool flush_cache             = arg_parser.get_bool("flush_cache");

    ck_tile::StreamKReductionStrategy reduction_strategy =
        get_reduction_strategy_value(arg_parser.get_str("reduction_strategy"));
    uint32_t num_sk_blocks = static_cast<uint32_t>(arg_parser.get_int("num_sk_blocks"));

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    auto [ave_time, num_wgs_per_tile] = invoke_gemm<GemmConfig,
                                                    ADataType,
                                                    BDataType,
                                                    ck_tile::tuple<>,
                                                    AccDataType,
                                                    CDataType,
                                                    ALayout,
                                                    BLayout,
                                                    ck_tile::tuple<>,
                                                    CLayout>(a_m_k_dev_buf,
                                                             b_k_n_dev_buf,
                                                             c_m_n_dev_buf,
                                                             M,
                                                             N,
                                                             K,
                                                             stride_A,
                                                             stride_B,
                                                             stride_C,
                                                             n_warmup,
                                                             n_repeat,
                                                             flush_cache,
                                                             reduction_strategy,
                                                             num_sk_blocks);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M=" << M << " N=" << N << " K=" << K
              << " StrideA=" << stride_A << " StrideB=" << stride_B << " StrideC=" << stride_C
              << " A_Layout=" << ALayout::name << " B_Layout=" << BLayout::name
              << " C_Layout=" << CLayout::name << " A_Type=" << DataTypeTraits<ADataType>::name
              << " B_Type=" << DataTypeTraits<BDataType>::name
              << " C_Type=" << DataTypeTraits<CDataType>::name
              << " reduction_strategy=" << arg_parser.get_str("reduction_strategy") << " "
              << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    bool pass = true;

    // Memory on host to store gpu reference result
    ck_tile::HostTensor<CDataType> c_m_n_ref(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
    c_m_n_ref.SetZero();

    if(arg_parser.get_int("v") == 1) // Validate on the CPU
    {
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_ref.mData.begin(), c_m_n_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, num_wgs_per_tile, max_accumulated_value);
        pass = do_verify(c_m_n_dev_result, c_m_n_ref, rtol_atol, "CPU");
    }
    else if(arg_parser.get_int("v") == 2) // Validate on the GPU
    {
        // Memory on device to store gpu reference result
        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_ref.get_element_space_size_in_bytes());
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* d_B = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* d_C = static_cast<CDataType*>(c_m_n_gpu_buf_ref.GetDeviceBuffer());

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_ref.data());

        const float max_accumulated_value =
            *std::max_element(c_m_n_ref.mData.begin(), c_m_n_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, num_wgs_per_tile, max_accumulated_value);
        pass = do_verify(c_m_n_dev_result, c_m_n_ref, rtol_atol, "GPU");
    }

    return pass;
}
