// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
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
    // Calculate error due to split_k accumulation
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
          bool Persistent,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm(int n_warmup,
                  int n_repeat,
                  int group_count,
                  const std::vector<grouped_gemm_kargs>& args)
{
    // Workspace memory allocated to hold the gemm descriptions.
    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(args));
    static constexpr ck_tile::index_t NumATensor = 1;
    static constexpr ck_tile::index_t NumBTensor = 1;
    static constexpr ck_tile::index_t NumDTensor = 2;
    float ave_time = 0;
    if constexpr(!Persistent)
    {
        // Regular version of grouped gemm
        ave_time = grouped_gemm<ADataType,
                                BDataType,
                                DsDataType,
                                AccDataType,
                                CDataType,
                                ALayout,
                                BLayout,
                                DsLayout,
                                CLayout,
                                CDEElementWise>(
            args,
            ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
            gemm_workspace.GetDeviceBuffer());
    }
    else
    {
        // NOTE: With the persistent TileLoop kernel, we do not necessarily need to have
        // the gemm problems known on the host. Instead, we can just pass the pointer
        // to the kernel and let the workgroups figure out which tiles to work on.
        // This is useful when the gemm problems are generated dynamically.
        // In this example however, we generate the `kargs` using the known gemm_descs,
        // and copy the gemm descriptions to the device memory.
        // The contents of the memory pointed to by `kargs_ptr` pointer could be
        // written by e.g. another kernel from earlier stage.
        std::vector<ck_tile::GemmTransKernelArg<NumDTensor>> kargs;
        void* kargs_ptr   = gemm_workspace.GetDeviceBuffer();
        const bool splitk = args[0].k_batch > 1;
        for(const auto& arg : args)
        {
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<NumATensor,NumBTensor,NumDTensor>{{arg.a_ptr},
                                                                  {arg.b_ptr},
                                                                  arg.ds_ptr,
                                                                  arg.e_ptr,
                                                                  arg.M,
                                                                  arg.N,
                                                                  arg.K,
                                                                  {arg.stride_A},
                                                                  {arg.stride_B},
                                                                  arg.stride_Ds,
                                                                  arg.stride_E,
                                                                  arg.k_batch});
        }
        const auto stream = ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat};
        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            kargs.size() * sizeof(ck_tile::GemmTransKernelArg<NumDTensor>),
                                            hipMemcpyHostToDevice,
                                            stream.stream_id_));
        ave_time = grouped_gemm_tileloop<GemmConfig,
                                         ALayout,
                                         BLayout,
                                         DsLayout,
                                         CLayout,
                                         ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccDataType,
                                         CDataType,
                                         CDEElementWise>(stream, group_count, kargs_ptr, splitk);
    }

    std::string op_name{"Grouped Gemm"};

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * args[j].M * args[j].N * args[j].K;

        num_btype += sizeof(ADataType) * args[j].M * args[j].K +
                     sizeof(BDataType) * args[j].K * args[j].N +
                     sizeof(CDataType) * args[j].M * args[j].N;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    return ave_time;
}

template <bool Persistent,
          typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename DsDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename D0Layout,
          typename D1Layout
        >
int run_grouped_gemm_example_with_layouts(int argc,
                                          char* argv[],
                                          const ALayout a_layout                  = ALayout{},
                                          const BLayout b_layout                  = BLayout{},
                                          [[maybe_unused]] const D0Layout d0_layout = D0Layout{},
                                          [[maybe_unused]] const D1Layout d1_layout = D1Layout{},
                                          [[maybe_unused]] const CLayout c_layout = CLayout{})
{

    auto [result, arg_parser] = create_args(argc, argv);

    if(!result)
    {
        return -1;
    };

    auto valid_input_data = [&](int group_count, const auto&... args) {
        return !(args.empty() || ...) && group_count == (args.size() == ...);
    };

    using DsLayout        = ck_tile::tuple<D0Layout, D1Layout>;
    using DssDataType        = ck_tile::tuple<DsDataType, DsDataType>;

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");
    const int kbatch      = arg_parser.get_int("kbatch");
    bool validate         = arg_parser.get_bool("validate");

    if(kbatch > 1 && validate && warmup + repeat > 1)
    {
        std::cout << "WARNING: Data validation enabled with SplitK and more than"
                  << "1 warmup/repeat. Disabling validation." << std::endl;
        validate = false;
    }

    std::vector<ck_tile::index_t> Ms        = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns        = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks        = arg_parser.get_int_vec("Ks");
    std::vector<ck_tile::index_t> stride_As = arg_parser.get_int_vec("stride_As");
    std::vector<ck_tile::index_t> stride_Bs = arg_parser.get_int_vec("stride_Bs");
    std::vector<ck_tile::index_t> stride_Cs = arg_parser.get_int_vec("stride_Cs");
    std::vector<ck_tile::index_t> stride_D0s = arg_parser.get_int_vec("stride_Dss");
    std::vector<ck_tile::index_t> stride_D1s = arg_parser.get_int_vec("stride_Dss");

    if(!valid_input_data(group_count, Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs))
    {
        std::cout << "Please check the input data. Default values will be used." << std::endl;
        for(int i = 0; i < group_count; i++)
        {
            Ms.push_back(256 + 256 * i);
            Ns.push_back(256 + 512 * i);
            Ks.push_back(512 + 128 * i);

            stride_As.push_back(Ks[i]);
            stride_Bs.push_back(Ks[i]);
            stride_Cs.push_back(Ns[i]);
            stride_D0s.push_back(Ns[i]);
            stride_D1s.push_back(Ns[i]);
        }
    }

    std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;
    std::vector<ck_tile::HostTensor<DsDataType>> d0_m_n_tensors;
    std::vector<ck_tile::HostTensor<DsDataType>> d1_m_n_tensors;

    a_m_k_tensors.reserve(group_count);
    b_k_n_tensors.reserve(group_count);
    c_m_n_tensors.reserve(group_count);
    d0_m_n_tensors.reserve(group_count);
    d1_m_n_tensors.reserve(group_count);

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> d0_m_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> d1_m_n_dev_buf;

    a_m_k_dev_buf.reserve(group_count);
    b_k_n_dev_buf.reserve(group_count);
    c_m_n_dev_buf.reserve(group_count);
    d0_m_n_dev_buf.reserve(group_count);
    d1_m_n_dev_buf.reserve(group_count);

    std::vector<grouped_gemm_kargs> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        const ck_tile::index_t M = Ms[i];
        const ck_tile::index_t N = Ns[i];
        const ck_tile::index_t K = Ks[i];

        stride_As[i] = ck_tile::get_default_stride(M, N, stride_As[i], is_row_major(a_layout));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));
        stride_Cs[i] = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));
        stride_D0s[i] = ck_tile::get_default_stride(M, N, stride_D0s[i], is_row_major(D0Layout{}));
        stride_D1s[i] = ck_tile::get_default_stride(M, N, stride_D1s[i], is_row_major(D1Layout{}));

        a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(a_layout))));
        b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(b_layout))));
        c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));
        d0_m_n_tensors.push_back(ck_tile::HostTensor<DsDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_D0s[i], is_row_major(D0Layout{}))));
        d1_m_n_tensors.push_back(ck_tile::HostTensor<DsDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_D1s[i], is_row_major(D1Layout{}))));

        std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                  << " b_k_n: " << b_k_n_tensors[i].mDesc << " c_m_n: " << c_m_n_tensors[i].mDesc
                  << std::endl;

        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);
        ck_tile::FillUniformDistribution<DsDataType>{-1.f, 1.f}(d0_m_n_tensors[i]);
        ck_tile::FillUniformDistribution<DsDataType>{-1.f, 1.f}(d1_m_n_tensors[i]);
        a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            a_m_k_tensors[i].get_element_space_size_in_bytes()));
        b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            b_k_n_tensors[i].get_element_space_size_in_bytes()));
        c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            c_m_n_tensors[i].get_element_space_size_in_bytes()));
        d0_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            d0_m_n_tensors[i].get_element_space_size_in_bytes()));
        d1_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            d1_m_n_tensors[i].get_element_space_size_in_bytes()));
        
        a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
        b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
        d0_m_n_dev_buf[i]->ToDevice(d0_m_n_tensors[i].data());
        d1_m_n_dev_buf[i]->ToDevice(d1_m_n_tensors[i].data());

        c_m_n_dev_buf[i]->SetZero();
        c_m_n_tensors[i].SetZero();
        
        const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
        const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
        void* p_c       = c_m_n_dev_buf[i]->GetDeviceBuffer();
        
        std::array<const void*, DssDataType::size()> ds_ptr_buf = {d0_m_n_dev_buf[i]->GetDeviceBuffer(),
                                                              d1_m_n_dev_buf[i]->GetDeviceBuffer()};
        std::array<ck_tile::index_t, DssDataType::size()> stridesDs = {stride_D0s[i], stride_D1s[i]};                                                      
        gemm_descs.push_back(
            {p_a, p_b, ds_ptr_buf, p_c, kbatch, M, N, K, stride_As[i], stride_Bs[i], stridesDs, stride_Cs[i]});
    }

    using CDElementWiseFn = ck_tile::element_wise::MultiDMultiply; // D0 for A scale; D1 for B scale
   
    invoke_gemm<GemmConfig,
                ADataType,
                BDataType,
                DssDataType,
                AccDataType,
                CDataType,
                ALayout,
                BLayout,
                DsLayout,
                CLayout,
                Persistent,
                CDElementWiseFn>(warmup, repeat, group_count, gemm_descs);

    for(int i = 0; i < group_count; i++)
    {
        c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
    }
    // for(int i = 0; i < 100; i++)
    // {
    //     printf("%f \n",ck_tile::type_convert<float>(c_m_n_tensors[0](0,i)));
    // }
    bool pass{true};
    if(validate)
    {
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
                Ms[i], Ns[i], stride_Cs[i], is_row_major(CLayout{})));
            c_m_n_host_ref.SetZero();

            ck_tile::reference_gemm_multiple_d<ADataType,
                                       BDataType,
                                       DssDataType,
                                       AccDataType,
                                       CDataType,
                                       CDElementWiseFn>(
            a_m_k_tensors[i], b_k_n_tensors[i], {d0_m_n_tensors[i], d1_m_n_tensors[i]}, c_m_n_host_ref);

            // ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            //     a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
            const float max_accumulated_value =
                *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                    Ks[i], kbatch, max_accumulated_value);
            pass &= ck_tile::check_err(c_m_n_tensors[i],
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));
            std::cout << "gemm[" << i
                      << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

template <bool Persistent, typename GemmConfig, typename PrecType>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;
    // Specific type aliases for easy access
    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using CDataType   = typename Types::CDataType;
    using DsDataType  = typename Types::DsDataType;
    if(a_layout == "R" && b_layout == "C")
    {
        return run_grouped_gemm_example_with_layouts<Persistent,
                                                     GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     DsDataType,
                                                     AccDataType>(argc, argv, Row{}, Col{}, Row{}, Row{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "R")
    {
        return run_grouped_gemm_example_with_layouts<Persistent,
                                                     GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     DsDataType,
                                                     AccDataType>(argc, argv, Row{}, Row{}, Row{}, Row{}, Row{});
    }
    else if(a_layout == "C" && b_layout == "R")
    {
        return run_grouped_gemm_example_with_layouts<Persistent,
                                                     GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     DsDataType,
                                                     AccDataType>(argc, argv, Col{}, Row{}, Row{}, Row{}, Row{});
    }
    else if(a_layout == "C" && b_layout == "C")
    {
        return run_grouped_gemm_example_with_layouts<Persistent,
                                                     GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     DsDataType,
                                                     AccDataType>(argc, argv, Col{}, Col{}, Row{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}

template <bool Persistent, template <typename PrecType> typename GemmConfig>
int run_grouped_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<Persistent, GemmConfig<ck_tile::half_t>, ck_tile::half_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<Persistent, GemmConfig<ck_tile::fp8_t>, ck_tile::fp8_t>(
            a_layout, b_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type configuration.");
    }
}
