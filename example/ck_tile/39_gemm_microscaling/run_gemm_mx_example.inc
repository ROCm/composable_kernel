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
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename AScaleLayout,
          typename BLayout,
          typename BScaleLayout,
          typename CLayout,
          uint32_t BlockScaleSize>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem& a_m_k_scale_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem& b_k_n_scale_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_AQ,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat)
{
    ck_tile::GemmMXKernelArgs args;
    args.a_ptr          = a_m_k_dev_buf.GetDeviceBuffer();
    args.a_scale_ptr_   = a_m_k_scale_dev_buf.GetDeviceBuffer();
    args.b_ptr          = b_k_n_dev_buf.GetDeviceBuffer();
    args.b_scale_ptr_   = b_k_n_scale_dev_buf.GetDeviceBuffer();
    args.c_ptr          = c_m_n_dev_buf.GetDeviceBuffer();
    args.k_batch        = kbatch;
    args.M              = M;
    args.N              = N;
    args.K              = K;
    args.stride_A       = stride_A;
    args.stride_scale_A = stride_scale_A; // stride for A scale
    args.stride_B       = stride_B;
    args.stride_scale_B = stride_scale_B; // stride for B scale
    args.stride_C       = stride_C;

    float ave_time = gemm_mx_calc<ADataType,
                                  AScaleDataType,
                                  BDataType,
                                  BScaleDataType,
                                  AccDataType,
                                  CDataType,
                                  BDataType,
                                  ALayout,
                                  BLayout,
                                  CLayout,
                                  BlockScaleSize>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::size_t flop = std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / BlockScaleSize;
    std::size_t num_byte =
        sizeof(ADataType) * M * K / ck_tile::numeric_traits<ADataType>::PackedSize +
        sizeof(BDataType) * K * N / ck_tile::numeric_traits<BDataType>::PackedSize +
        sizeof(ck_tile::e8m0_bexp_t) * M * K / BlockScaleSize +
        sizeof(ck_tile::e8m0_bexp_t) * K * N / BlockScaleSize + sizeof(CDataType) * M * N;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideScaleA =" << stride_scale_A
              << " StrideB =" << stride_B << " StrideScaleB =" << stride_scale_B
              << " StrideC =" << stride_C << " A_Layout =" << ALayout::name
              << " B_Layout =" << BLayout::name << " C_Layout =" << CLayout::name
              << " A_Type = " << DataTypeTraits<ADataType>::name
              << " A_Scale_Type = " << DataTypeTraits<AScaleDataType>::name
              << " B_Type = " << DataTypeTraits<BDataType>::name
              << " B_Scale_Type = " << DataTypeTraits<BScaleDataType>::name
              << " Acc_Type = " << DataTypeTraits<AccDataType>::name
              << " C_Type = " << DataTypeTraits<CDataType>::name << " : " << ave_time << " ms, "
              << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

template <typename TypeConfig,
          uint32_t BlockScaleSize,
          typename ALayout,
          typename AScaleLayout,
          typename BLayout,
          typename BScaleLayout,
          typename CLayout>
int run_gemm_example_with_layouts(int argc,
                                  char* argv[],
                                  const ALayout a_layout                  = ALayout{},
                                  const AScaleLayout a_scale_layout       = AScaleLayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  const BScaleLayout b_scale_layout       = BScaleLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType       = typename TypeConfig::ADataType;
    using AScaleDataType  = typename TypeConfig::ScaleDataType;
    using BDataType       = typename TypeConfig::BDataType;
    using BScaleDataType  = typename TypeConfig::ScaleDataType;
    using XPackedDataType = typename TypeConfig::ScalePackDataType;
    using AccDataType     = typename TypeConfig::AccDataType;
    using CDataType       = typename TypeConfig::CDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    if(K % BlockScaleSize != 0)
    {
        throw std::runtime_error("K must be aligned with BlockScaleSize");
    }
    if(K % ck_tile::packed_size_v<ADataType> != 0 || K % ck_tile::packed_size_v<BDataType> != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of packed size.");
    }

    ck_tile::index_t AQK = K / QuantGroupSize;

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    auto Scale_Padded_M = ck::math::integer_least_multiple(M, ScaleBlockSize);
    auto Scale_Stride_A = ck_tile::get_default_stride(
        Scale_Padded_M, K / ScaleBlockSize, 0, is_row_major(a_scale_layout));
    auto Scale_Stride_B =
        ck_tile::get_default_stride(K / ScaleBlockSize, N, 0, is_row_major(b_scale_layout));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));

    // B Shuffled tensor
    // ck_tile::HostTensor<BDataType> b_k_n_shuffled(
    //     ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));

    // A, B scale tensors
    ck_tile::HostTensor<AScaleDataType> a_m_k_scale(ck_tile::host_tensor_descriptor(
        Scale_Padded_M, K / ScaleBlockSize, Scale_Stride_A, is_row_major(a_scale_layout)));
    ck_tile::HostTensor<BScaleDataType> b_k_n_scale(ck_tile::host_tensor_descriptor(
        K / ScaleBlockSize, N, Scale_Stride_B, is_row_major(b_scale_layout)));
    // A, B scale shuffled tensors
    ck_tile::HostTensor<AScaleDataType> a_m_k_scale_shuffled(ck_tile::host_tensor_descriptor(
        Scale_Padded_M, K / ScaleBlockSize, Scale_Stride_A, is_row_major(a_scale_layout)));
    ck_tile::HostTensor<BScaleDataType> b_k_n_scale_shuffled(ck_tile::host_tensor_descriptor(
        K / ScaleBlockSize, N, Scale_Stride_B, is_row_major(b_scale_layout)));

    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint32_t> fill_seed(0, 500);

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-1.0f, 1.0f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-1.0f, 1.0f}(b_k_n);
        ck_tile::FillUniformDistribution<ASCaleDataType>{-1.0f, 1.0f}(a_m_k_scale);
        ck_tile::FillUniformDistribution<BScaleDataType>{-1.0f, 1.0f}(b_k_n_scale);
    }
    else if(init_method == 1)
    {
        ck_tile::FillConstant<ADataType>{1.0f, 1.0f}(a_m_k);
        ck_tile::FillConstant<BDataType>{1.0f, 1.0f}(b_k_n);
        ck_tile::FillConstant<ASCaleDataType>{1.0f, 1.0f}(a_m_k_scale);
        ck_tile::FillConstant<BScaleDataType>{1.0f, 1.0f}(b_k_n_scale);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
        a_m_k_scale.SetZero();
        b_k_n_scale.SetZero();
    }

    // Shuffle A, B scale tensors
    preShuffleScaleBuffer<ck_tile::is_same_v<a_layout, ck_tile::tensor_layout::gemm::RowMajor>>(
        a_m_k_scale.data(), a_m_k_scale_shuffled.data(), Scale_Padded_M, K / ScaleBlockSize);
    preShuffleScaleBuffer<ck_tile::is_same_v<b_layout, ck_tile::tensor_layout::gemm::ColumnMajor>>(
        b_k_n_scale.data(), b_k_n_scale_shuffled.data(), N, K / ScaleBlockSize);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem a_m_k_scale_dev_buf(a_m_k_scale.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_scale_dev_buf(b_k_n_scale.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    a_m_k_scale_dev_buf.ToDevice(a_m_k_scale.data());
    b_k_n_scale_dev_buf.ToDevice(b_k_n_scale.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    invoke_gemm<ADataType,
                XPackedDataType,
                BDataType,
                XPackedDataType,
                AccDataType,
                CDataType,
                ALayout,
                AScaleLayout,
                BLayout,
                BScaleLayout,
                CLayout,
                BlockScaleSize>(a_m_k_dev_buf,
                                a_m_k_scale_dev_buf,
                                b_k_n_dev_buf,
                                b_k_n_scale_dev_buf,
                                c_m_n_dev_buf,
                                M,
                                N,
                                K,
                                stride_A,
                                Scale_Stride_A,
                                stride_B,
                                Scale_Stride_B,
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
