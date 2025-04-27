// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

using F16 = ck_tile::half_t;
// using BF16 = ck::bhalf_t;
using F8  = ck_tile::fp8_t;
using F32 = float;

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t, std::string mfma_dtype, int mfma_type = 0)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    if((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 0)
    {
        ck_tile::HostTensor<T> t_view({n_ / 32, 32, k_ / 16, 2, 8});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else if((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 1)
    {
        ck_tile::HostTensor<T> t_view({n_ / 16, 16, k_ / 32, 4, 8});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else if((mfma_dtype == "int8" || mfma_dtype == "fp8") && mfma_type == 0)
    {
        ck_tile::HostTensor<T> t_view({n_ / 32, 32, k_ / 32, 2, 16});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else if((mfma_dtype == "int8" || mfma_dtype == "fp8") && mfma_type == 1)
    {
        ck_tile::HostTensor<T> t_view({n_ / 16, 16, k_ / 64, 4, 16});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    return t;
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

template <typename ALayout, typename BLayout, typename CLayout>
float invoke_moe_gemm(int n_warmup, int n_repeat, const moe_gemm_kargs& args)
{
    float ave_time = moe_gemm<ALayout, BLayout, CLayout>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::string op_name{"Moe Gemm"};

    std::size_t flop = 0, num_btype = 0;

    flop += std::size_t(2) * args.M * args.N * args.K;

    num_btype += sizeof(ADataType) * args.M * args.K + sizeof(BDataType) * args.K * args.N +
                 sizeof(CDataType) * args.M * args.N;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    return ave_time;
}

template <typename ALayout, typename BLayout, typename CLayout>
int run_moe_gemm_example_with_layouts(int argc,
                                      char* argv[],
                                      const ALayout a_layout                  = ALayout{},
                                      const BLayout b_layout                  = BLayout{},
                                      [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);

    if(!result)
    {
        return -1;
    };

    // auto valid_input_data = [&](int group_count, const auto&... args) {
    //     return !(args.empty() || ...) && group_count == (args.size() == ...);
    // };

    // ck_tile::index_t N               = 4096;
    // ck_tile::index_t K               = 4096;
    // ck_tile::index_t experts         = 8;
    // ck_tile::index_t sorted_tile_num = 8;
    // ck_tile::index_t valid_tile_num  = 8;
    // ck_tile::index_t tokens          = 128;
    // ck_tile::index_t topk            = 2;

    const ck_tile::index_t N          = arg_parser.get_int("N");
    const ck_tile::index_t K          = arg_parser.get_int("K");
    ck_tile::index_t stride_A         = arg_parser.get_int("stride_A");
    ck_tile::index_t stride_B         = arg_parser.get_int("stride_B");
    ck_tile::index_t stride_C         = arg_parser.get_int("stride_C");
    const ck_tile::index_t num_tokens = arg_parser.get_int("NumTokens");
    const ck_tile::index_t topk       = arg_parser.get_int("TopK");
    const ck_tile::index_t repeat     = arg_parser.get_int("repeat");
    const ck_tile::index_t experts    = arg_parser.get_int("experts");

    // TODO: replace the magic declaration
    const ck_tile::index_t MPerBlock = 128;
    // const ck_tile::index_t max_num_tokens_padded = topk * num_tokens + experts * MPerBlock - topk;
    // const ck_tile::index_t max_num_m_blocks = (max_num_tokens_padded + MPerBlock - 1) / MPerBlock;

    ck_tile::index_t sorted_tile_num = 8;
    ck_tile::index_t valid_tile_num  = sorted_tile_num;
    // ck_tile::index_t sorted_tile_num = 16;
    // ck_tile::index_t valid_tile_num  = 13;
    ck_tile::index_t sorted_size     = sorted_tile_num * MPerBlock;
    // ck_tile::index_t valid_size      = valid_tile_num * MPerBlock;

    const ck_tile::index_t M = sorted_tile_num * MPerBlock;
    // const ck_tile::index_t M = max_num_tokens_padded;

    std::unique_ptr<ck_tile::DeviceMem> a_m_k_dev_buf;
    std::unique_ptr<ck_tile::DeviceMem> b_origin_dev_buf;
    std::unique_ptr<ck_tile::DeviceMem> b_shuffle_dev_buf;
    std::unique_ptr<ck_tile::DeviceMem> c_m_n_dev_buf;

    stride_A = ck_tile::get_default_stride(num_tokens, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(num_tokens * topk, N, stride_C, is_row_major(CLayout{}));

    auto a_m_k_tensor = ck_tile::HostTensor<ADataType>(
        ck_tile::host_tensor_descriptor(num_tokens, K, stride_A, is_row_major(a_layout)));

    // TODO: add the experts' weights in b
    auto b_k_n_tensor = ck_tile::HostTensor<BDataType>(
        is_row_major(b_layout)
            ? ck_tile::host_tensor_descriptor(experts * N, K, stride_B, is_row_major(b_layout))
            : ck_tile::host_tensor_descriptor(K, experts * N, stride_B, is_row_major(b_layout)));

    std::string mfma    = arg_parser.get_str("prec");

    auto c_m_n_tensor = ck_tile::HostTensor<CDataType>(
        ck_tile::host_tensor_descriptor(num_tokens * topk, N, stride_C, is_row_major(CLayout{})));

    ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensor);
    ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensor);

    auto b_shuffle_host = shuffle_b(b_k_n_tensor, mfma, 0);

    std::cout << "gemm"
              << " a_m_k: " << a_m_k_tensor.mDesc << " b_k_n: " << b_k_n_tensor.mDesc
              << " b_shuffle: " << b_shuffle_host.mDesc << " c_m_n: " << c_m_n_tensor.mDesc << std::endl;

    a_m_k_dev_buf =
        std::make_unique<ck_tile::DeviceMem>(a_m_k_tensor.get_element_space_size_in_bytes());
    b_origin_dev_buf =
        std::make_unique<ck_tile::DeviceMem>(b_k_n_tensor.get_element_space_size_in_bytes());
    b_shuffle_dev_buf =
        std::make_unique<ck_tile::DeviceMem>(b_shuffle_host.get_element_space_size_in_bytes());
    c_m_n_dev_buf =
        std::make_unique<ck_tile::DeviceMem>(c_m_n_tensor.get_element_space_size_in_bytes());

    a_m_k_dev_buf->ToDevice(a_m_k_tensor.data());
    b_origin_dev_buf->ToDevice(b_k_n_tensor.data());
    b_shuffle_dev_buf->ToDevice(b_shuffle_host.data());
    c_m_n_dev_buf->SetZero();
    c_m_n_tensor.SetZero();

    const void* p_a = a_m_k_dev_buf->GetDeviceBuffer();
    const void* p_b_origin = b_origin_dev_buf->GetDeviceBuffer();
    const void* p_b_shuffle = b_shuffle_dev_buf->GetDeviceBuffer();
    void* p_c       = c_m_n_dev_buf->GetDeviceBuffer();

    // TODO: malloc and init sorted tokens and max tokens buffer

    ck_tile::HostTensor<ck_tile::index_t> expert_ids(
        ck_tile::HostTensorDescriptor({sorted_tile_num}, {1}));
    ck_tile::HostTensor<ck_tile::index_t> sorted_token_ids(
        ck_tile::HostTensorDescriptor({sorted_size}, {1}));
    ck_tile::HostTensor<ck_tile::index_t> max_token_id(
        ck_tile::HostTensorDescriptor({1 + sorted_tile_num}));

    std::unique_ptr<ck_tile::DeviceMem> sorted_token_ids_dev = std::make_unique<ck_tile::DeviceMem>(
        sizeof(ck_tile::index_t) * sorted_token_ids.get_element_space_size_in_bytes());
    std::unique_ptr<ck_tile::DeviceMem> expert_ids_dev = std::make_unique<ck_tile::DeviceMem>(
        sizeof(ck_tile::index_t) * expert_ids.get_element_space_size_in_bytes());
    std::unique_ptr<ck_tile::DeviceMem> max_token_id_dev = std::make_unique<ck_tile::DeviceMem>(
        sizeof(ck_tile::index_t) * max_token_id.get_element_space_size_in_bytes());

    max_token_id.mData = {valid_tile_num * MPerBlock, 0, 1, 2, 3, 4, 6, 7, 8, 8};
    int eids[]         = {0, 1, 2, 3, 4, 4, 5, 6, 3, 3, 3, 3}; // {2, 1, 1, 2, 2, 2, 1, 2}
    // max_token_id.mData = {valid_size, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 0, 0};
    // int eids[]         = {0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 3, 3, 3};
    for(int i = 0; i < sorted_tile_num; i++)
    {
        expert_ids.mData[i] = eids[i];
    }
    // int token_per_tile = (num_tokens * topk + valid_tile_num - 1) / valid_tile_num;
    int token_per_tile = num_tokens * topk / valid_tile_num;
    int tokenid        = 0;
    // sorted_token_ids.mData[0] = 0;
    for(int i = 0; i < sorted_tile_num * MPerBlock; i++)
    {
        int tile_off = i % MPerBlock;
        if(tile_off < token_per_tile && tokenid < num_tokens * topk)
        {
            sorted_token_ids.mData[i] = (tokenid % num_tokens) | ((tokenid / num_tokens) << 24);
            tokenid++;
        }
        else
        {
            sorted_token_ids.mData[i] = num_tokens - 1;
        }
    }

    sorted_token_ids_dev->ToDevice(sorted_token_ids.data());
    expert_ids_dev->ToDevice(expert_ids.data());
    max_token_id_dev->ToDevice(max_token_id.data());

    const ck_tile::index_t* p_sorted_token_ids_dev =
        static_cast<ck_tile::index_t*>(sorted_token_ids_dev->GetDeviceBuffer());
    const ck_tile::index_t* p_expert_ids_dev =
        static_cast<ck_tile::index_t*>(expert_ids_dev->GetDeviceBuffer());
    const ck_tile::index_t* p_max_token_id_dev =
        static_cast<ck_tile::index_t*>(max_token_id_dev->GetDeviceBuffer());

    moe_gemm_kargs gemm_desc{p_sorted_token_ids_dev,
                             p_expert_ids_dev,
                             p_max_token_id_dev,
                             p_a,
                             p_b_shuffle,
                             p_c,
                             num_tokens,
                             topk,
                             1, //k_batch
                             M,
                             N,
                             K,
                             stride_A,
                             stride_B,
                             stride_C};

    invoke_moe_gemm<ALayout, BLayout, CLayout>(3, repeat, gemm_desc);

    c_m_n_dev_buf->FromDevice(c_m_n_tensor.data());

    bool pass{true};
    if(arg_parser.get_int("validate"))
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(num_tokens * topk, N, stride_C, is_row_major(CLayout{})));

        c_m_n_host_ref.SetZero();

        std::unique_ptr<ck_tile::DeviceMem> c_m_n_ref_buf =
            std::make_unique<ck_tile::DeviceMem>(c_m_n_tensor.get_element_space_size_in_bytes());

        c_m_n_ref_buf->SetZero();

        ck_tile::reference_moe_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout>(
            p_sorted_token_ids_dev,
            p_expert_ids_dev,
            p_max_token_id_dev,
            static_cast<const ADataType*>(p_a),
            static_cast<const BDataType*>(p_b_origin),
            static_cast<CDataType*>(c_m_n_ref_buf->GetDeviceBuffer()),
            num_tokens,
            topk,
            M,
            N,
            K,
            stride_A,
            stride_B,
            stride_C);
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, 1 /*kbatch*/, max_accumulated_value);
        c_m_n_ref_buf->FromDevice(c_m_n_host_ref.data());

        // for(int im = 0; im < M; im++)
        // {
        //     for(int in = 0; in < N; in++)
        //     {
        //         // if (static_cast<float>(static_cast<CDataType*>(p_c)[im * N + in]) != 0)
        //         printf("c[%d][%d]: %f ",
        //                im,
        //                in,
        //                static_cast<float>(static_cast<CDataType*>(p_c)[im * N + in]));
        //         printf("ref[%d][%d]: %f \n",
        //                im,
        //                in,
        //                static_cast<float>(
        //                    static_cast<CDataType*>(c_m_n_host_ref.data())[im * N + in]));
        //     }
        // }

        pass = ck_tile::check_err(c_m_n_tensor,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;

        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

int run_moe_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout = arg_parser.get_str("a_layout");
    const std::string b_layout = arg_parser.get_str("b_layout");

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        return run_moe_gemm_example_with_layouts(argc, argv, Row{}, Col{}, Row{});
    }
    // else if(a_layout == "R" && b_layout == "R")
    // {
    //     return run_grouped_gemm_example_with_layouts(argc, argv, Row{}, Row{}, Row{});
    // }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}
