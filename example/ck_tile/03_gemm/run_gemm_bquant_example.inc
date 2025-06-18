// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <bit>
#include <random>

// mfma_type, 0:32x32, 1:16x16
template <typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t, std::string mfma_dtype, int mfma_type)
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
    else if((mfma_dtype == "int8" || mfma_dtype == "fp8" || mfma_dtype == "bf8") && mfma_type == 0)
    {
        ck_tile::HostTensor<T> t_view({n_ / 32, 32, k_ / 32, 2, 16});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else if((mfma_dtype == "int8" || mfma_dtype == "fp8" || mfma_dtype == "bf8") && mfma_type == 1)
    {
        ck_tile::HostTensor<T> t_view({n_ / 16, 16, k_ / 64, 4, 16});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    return t;
}

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

template <typename ADataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename CLayout,
          uint32_t QuantGroupSize,
          bool as_br_cr>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem& bq_m_aqk_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t AQK,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_BQ,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat)
{
    ck_tile::BQuantGemmHostArgs args;
    args.a_ptr     = a_m_k_dev_buf.GetDeviceBuffer();
    args.bq_ptr    = bq_m_aqk_dev_buf.GetDeviceBuffer();
    args.b_ptr     = b_k_n_dev_buf.GetDeviceBuffer();
    args.c_ptr     = c_m_n_dev_buf.GetDeviceBuffer();
    args.k_batch   = kbatch;
    args.M         = M;
    args.N         = N;
    args.K         = K;
    args.QK        = AQK;
    args.stride_A  = stride_A;
    args.stride_B  = stride_B;
    args.stride_C  = stride_C;
    args.stride_BQ = stride_BQ;

    float ave_time = gemm_calc_bquant<ADataType,
                                      BDataType,
                                      BQDataType,
                                      AccDataType,
                                      CDataType,
                                      ADataType,
                                      ALayout,
                                      BLayout,
                                      CLayout,
                                      QuantGroupSize,
                                      as_br_cr>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::size_t flop     = std::size_t(2) * M * N * K;
    std::size_t num_byte = sizeof(ADataType) * M * K + sizeof(BQDataType) * M * AQK +
                           sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideBQ =" << stride_BQ << " StrideB =" << stride_B
              << " StrideC =" << stride_C << " A_Layout =" << ALayout::name
              << " B_Layout =" << BLayout::name << " C_Layout =" << CLayout::name
              << " A_Type = " << DataTypeTraits<ADataType>::name
              << " BQ_Type = " << DataTypeTraits<BQDataType>::name
              << " B_Type = " << DataTypeTraits<BDataType>::name
              << " Acc_Type = " << DataTypeTraits<AccDataType>::name
              << " C_Type = " << DataTypeTraits<CDataType>::name << " : " << ave_time << " ms, "
              << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

template <typename TypeConfig,
          uint32_t QuantGroupSize,
          bool as_br_cr,
          typename ALayout,
          typename BQLayout,
          typename BLayout,
          typename CLayout>
int run_gemm_example_with_layouts(int argc,
                                  char* argv[],
                                  const ALayout a_layout                  = ALayout{},
                                  const BQLayout bq_layout                = BQLayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType = typename TypeConfig::ADataType;
    using BQDataType = typename TypeConfig::QDataType;
    using BDataType = typename TypeConfig::BDataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using CDataType = typename TypeConfig::CDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    assert(K % QuantGroupSize == 0 && "K must be aligned with QuantGroupSize");
    ck_tile::index_t BQK = K / QuantGroupSize;

    ck_tile::index_t stride_A  = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_BQ = arg_parser.get_int("stride_q");
    ck_tile::index_t stride_B  = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C  = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");

    stride_A  = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_BQ = ck_tile::get_default_stride(BQK, N, stride_BQ, is_row_major(bq_layout));
    stride_B  = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C  = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BQDataType> bq_m_bqk(
        ck_tile::host_tensor_descriptor(BQK, N, stride_BQ, is_row_major(bq_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint32_t> fill_seed(0, 500);

    if(init_method == 0)
    {
        if(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            constexpr auto int4_array = std::array<uint8_t, 16>{0x77,
                                                                0x66,
                                                                0x55,
                                                                0x44,
                                                                0x33,
                                                                0x22,
                                                                0x11,
                                                                0x00,
                                                                0xff,
                                                                0xee,
                                                                0xdd,
                                                                0xcc,
                                                                0xbb,
                                                                0xaa,
                                                                0x99,
                                                                0x88};
            std::uniform_int_distribution<std::uint32_t> dis(0, 15);
            for(size_t i = 0; i < b_k_n.size(); i++)
            {
                int randomInt   = dis(gen);
                b_k_n.data()[i] = int4_array[randomInt];
            }
        }
        else
        {
            ck_tile::FillUniformDistribution<BDataType>{-2.0f, 3.0f, fill_seed(gen)}(b_k_n);
        }
        ck_tile::FillUniformDistribution<BQDataType>{-2.0f, 2.0f, fill_seed(gen)}(bq_m_bqk);
        ck_tile::FillUniformDistribution<ADataType>{-5.0f, 5.0f, fill_seed(gen)}(a_m_k);
    }
    else if(init_method == 1)
    {
        std::cout << "Monotonic initialization is not supported." << std::endl;
        return 0;
    }
    else if(init_method == 2)
    {
        ck_tile::FillConstant<ADataType>{static_cast<ADataType>(0x38)}(a_m_k);
        ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x22)}(b_k_n);
        ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(0.5f)}(bq_m_bqk);
    }
    else
    {
        a_m_k.SetZero();
        bq_m_bqk.SetZero();
        b_k_n.SetZero();
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem bq_m_bqk_dev_buf(bq_m_bqk.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    if constexpr(as_br_cr) {
        // do pre-shuffle
        std::string mfma = arg_parser.get_str("prec");
#if defined(USING_MFMA_16x16x32) && defined(ENABLE_FP8)
        ck_tile::index_t mfma_type = 1;
#else
        ck_tile::index_t mfma_type = 0;
#endif
        ck_tile::HostTensor<BDataType> b_k_n_shuffled = shuffle_b(b_k_n, mfma, mfma_type);

        b_k_n_dev_buf.Realloc(b_k_n_shuffled.get_element_space_size_in_bytes());
        b_k_n_dev_buf.ToDevice(b_k_n_shuffled.data());
    } else {
        b_k_n_dev_buf.ToDevice(b_k_n.data());
    }

    static_assert(!GemmConfig::PermuteA, "Not implemented");
    static_assert(!GemmConfig::PermuteB, "Not implemented");

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    bq_m_bqk_dev_buf.ToDevice(bq_m_bqk.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    invoke_gemm<ADataType,
                BDataType,
                BQDataType,
                AccDataType,
                CDataType,
                ALayout,
                BQLayout,
                BLayout,
                CLayout,
                QuantGroupSize,
                as_br_cr>(a_m_k_dev_buf,
                          bq_m_bqk_dev_buf,
                          b_k_n_dev_buf,
                          c_m_n_dev_buf,
                          M,
                          N,
                          K,
                          BQK,
                          stride_A,
                          stride_BQ,
                          stride_B,
                          stride_C,
                          kbatch,
                          n_warmup,
                          n_repeat);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm_quant<ADataType,
                                      BQDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      QuantGroupSize, false>(a_m_k, bq_m_bqk, b_k_n, c_m_n_host_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        if(!pass)
        {
            std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "CPU verification " << (pass ? "Passed!" : "Failed ...") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        std::cout << "GPU verification is not implemented yet. Re-run with -v=1" << std::endl;
        return false;
    }

    return pass;
}
