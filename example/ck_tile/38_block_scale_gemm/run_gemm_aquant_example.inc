// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <bit>
#include <random>

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename CLayout,
          uint32_t QuantGroupSize>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem& aq_m_aqk_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t AQK,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_AQ,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat)
{
    ck_tile::AQuantGemmHostArgs args;
    args.a_ptr     = a_m_k_dev_buf.GetDeviceBuffer();
    args.aq_ptr    = aq_m_aqk_dev_buf.GetDeviceBuffer();
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
    args.stride_AQ = stride_AQ;

    float ave_time = gemm_calc_aquant<ADataType,
                                      AQDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      BDataType,
                                      ALayout,
                                      BLayout,
                                      CLayout,
                                      QuantGroupSize>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::size_t flop     = std::size_t(2) * M * N * K;
    std::size_t num_byte = sizeof(ADataType) * M * K + sizeof(AQDataType) * M * AQK +
                           sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideAQ =" << stride_AQ << " StrideB =" << stride_B
              << " StrideC =" << stride_C << " A_Layout =" << ALayout::name
              << " B_Layout =" << BLayout::name << " C_Layout =" << CLayout::name
              << " A_Type = " << DataTypeTraits<ADataType>::name
              << " AQ_Type = " << DataTypeTraits<AQDataType>::name
              << " B_Type = " << DataTypeTraits<BDataType>::name
              << " Acc_Type = " << DataTypeTraits<AccDataType>::name
              << " C_Type = " << DataTypeTraits<CDataType>::name << " : " << ave_time << " ms, "
              << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

template <typename TypeConfig,
          uint32_t QuantGroupSize,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename CLayout>
int run_gemm_example_with_layouts(int argc,
                                  char* argv[],
                                  const ALayout a_layout                  = ALayout{},
                                  const AQLayout aq_layout                = AQLayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType   = typename TypeConfig::ADataType;
    using AQDataType  = typename TypeConfig::QDataType;
    using BDataType   = typename TypeConfig::BDataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using CDataType   = typename TypeConfig::CDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    if(K % QuantGroupSize != 0)
    {
        throw std::runtime_error("K must be aligned with QuantGroupSize");
    }

    ck_tile::index_t AQK = K / QuantGroupSize;

    ck_tile::index_t stride_A  = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_AQ = arg_parser.get_int("stride_q");
    ck_tile::index_t stride_B  = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C  = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");

    stride_A  = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_AQ = ck_tile::get_default_stride(M, AQK, stride_AQ, is_row_major(aq_layout));
    stride_B  = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C  = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<AQDataType> aq_m_aqk(
        ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, is_row_major(aq_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint32_t> fill_seed(0, 500);

    if(init_method == 0)
    {
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            ck_tile::FillUniformDistribution<ck_tile::pk_int4_t>{-5.0f, 5.0f, fill_seed(gen)}(
                a_m_k);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f, fill_seed(gen)}(a_m_k);
        }
        ck_tile::FillUniformDistribution<AQDataType>{-2.0f, 2.0f, fill_seed(gen)}(aq_m_aqk);
        ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f, fill_seed(gen)}(b_k_n);
    }
    else if(init_method == 1)
    {
        std::cout << "Monotonic initialization is not supported." << std::endl;
        return 0;
    }
    else if(init_method == 2)
    {
        ck_tile::FillConstant<ADataType>{static_cast<ADataType>(0x22)}(a_m_k);
        ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(0.5f)}(aq_m_aqk);
        ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x38)}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        aq_m_aqk.SetZero();
        b_k_n.SetZero();
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem aq_m_aqk_dev_buf(aq_m_aqk.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    invoke_gemm<ADataType,
                AQDataType,
                BDataType,
                AccDataType,
                CDataType,
                ALayout,
                AQLayout,
                BLayout,
                CLayout,
                QuantGroupSize>(a_m_k_dev_buf,
                                aq_m_aqk_dev_buf,
                                b_k_n_dev_buf,
                                c_m_n_dev_buf,
                                M,
                                N,
                                K,
                                AQK,
                                stride_A,
                                stride_AQ,
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
                                      AQDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      QuantGroupSize,
                                      true>(a_m_k, aq_m_aqk, b_k_n, c_m_n_host_ref);
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
