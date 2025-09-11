// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include "contraction_utils.hpp"

template <typename ADataType, typename BDataType, typename EDataType, typename AccDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EDataType, EDataType, EDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType, EDataType, EDataType>(
        max_accumulated_value, kbatch);

    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_batched_contraction(ck_tile::DeviceMem& a_m_k_dev_buf,
                                 ck_tile::DeviceMem& b_k_n_dev_buf,
                                 ck_tile::DeviceMem& e_m_n_dev_buf,
                                 ck_tile::index_t M,
                                 ck_tile::index_t N,
                                 ck_tile::index_t K,
                                 const std::vector<ck_tile::index_t>& G_lengths,
                                 ck_tile::index_t stride_A,
                                 ck_tile::index_t stride_B,
                                 ck_tile::index_t stride_E,
                                 const std::vector<ck_tile::index_t>& batch_strides_A,
                                 const std::vector<ck_tile::index_t>& batch_strides_B,
                                 const std::vector<ck_tile::index_t>& batch_strides_E,
                                 ck_tile::index_t kbatch,
                                 int n_warmup,
                                 int n_repeat)
{
    ck_tile::BatchedContractionHostArgs<0> args{
        a_m_k_dev_buf.GetDeviceBuffer(), // a_ptr
        b_k_n_dev_buf.GetDeviceBuffer(), // b_ptr
        {},                              // ds_ptr (empty for NumDTensor=0)
        e_m_n_dev_buf.GetDeviceBuffer(), // e_ptr
        kbatch,                          // k_batch
        M,                               // M
        N,                               // N
        K,                               // K
        G_lengths,                       // G (G dimensions)
        stride_A,                        // stride_A
        stride_B,                        // stride_B
        {},                              // stride_Ds (empty)
        stride_E,                        // stride_E
        batch_strides_A,                 // batch_strides_A
        batch_strides_B,                 // batch_strides_B
        {},                              // batch_strides_Ds (empty)
        batch_strides_E                  // batch_strides_E
    };

    float ave_time = batched_contraction<ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccDataType,
                                         EDataType,
                                         ALayout,
                                         BLayout,
                                         DsLayout,
                                         ELayout,
                                         CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    return ave_time;
}

template <typename ALayout, typename BLayout, typename ELayout>
int run_batched_contraction_example_with_layouts(
    int argc,
    char* argv[],
    const ALayout a_layout                  = ALayout{},
    const BLayout b_layout                  = BLayout{},
    [[maybe_unused]] const ELayout e_layout = ELayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    std::string g_dims_str               = arg_parser.get_str("g_dims");
    std::vector<ck_tile::index_t> G_dims = parse_g_dimensions(g_dims_str);
    ck_tile::index_t total_G             = calculate_total_g(G_dims);

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_E = arg_parser.get_int("stride_e");

    ck_tile::index_t kbatch = arg_parser.get_int("split_k");

    int n_warmup = arg_parser.get_int("warmup");
    int n_repeat = arg_parser.get_int("repeat");

    using namespace ck_tile::literals;

    auto f_host_tensor_descriptor = [](std::size_t g,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return ck_tile::HostTensorDescriptor({g, row, col}, {batch_stride, stride, 1_uz});
        }
        else
        {
            return ck_tile::HostTensorDescriptor({g, row, col}, {batch_stride, 1_uz, stride});
        }
    };

    auto f_get_default_stride = [](std::size_t row,
                                   std::size_t col,
                                   std::size_t stride,
                                   auto layout) {
        if(stride == 0)
        {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return col;
            }
            else
            {
                return row;
            }
        }
        else
            return stride;
    };

    stride_A = f_get_default_stride(M, K, stride_A, a_layout);
    stride_B = f_get_default_stride(K, N, stride_B, b_layout);
    stride_E = f_get_default_stride(M, N, stride_E, e_layout);

    std::vector<ck_tile::index_t> G_strides_A = calculate_g_strides(G_dims, M * K);
    std::vector<ck_tile::index_t> G_strides_B = calculate_g_strides(G_dims, K * N);
    std::vector<ck_tile::index_t> G_strides_E = calculate_g_strides(G_dims, M * N);

    std::cout << "Debug Info:" << std::endl;
    std::cout << "G_dims: [";
    for(size_t i = 0; i < G_dims.size(); ++i)
    {
        std::cout << G_dims[i];
        if(i < G_dims.size() - 1)
            std::cout << ",";
    }
    std::cout << "]" << std::endl;
    std::cout << "total_G: " << total_G << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "stride_A=" << stride_A << ", stride_B=" << stride_B << ", stride_E=" << stride_E
              << std::endl;

    std::cout << "G_strides_A: [";
    for(size_t i = 0; i < G_strides_A.size(); ++i)
    {
        std::cout << G_strides_A[i];
        if(i < G_strides_A.size() - 1)
            std::cout << ",";
    }
    std::cout << "]" << std::endl;

    std::cout << "G_strides_B: [";
    for(size_t i = 0; i < G_strides_B.size(); ++i)
    {
        std::cout << G_strides_B[i];
        if(i < G_strides_B.size() - 1)
            std::cout << ",";
    }
    std::cout << "]" << std::endl;

    std::cout << "G_strides_E: [";
    for(size_t i = 0; i < G_strides_E.size(); ++i)
    {
        std::cout << G_strides_E[i];
        if(i < G_strides_E.size() - 1)
            std::cout << ",";
    }
    std::cout << "]" << std::endl;

    ck_tile::HostTensor<::ADataType> a_m_k(
        f_host_tensor_descriptor(total_G, M, K, stride_A, M * K, a_layout));
    ck_tile::HostTensor<::BDataType> b_k_n(
        f_host_tensor_descriptor(total_G, K, N, stride_B, K * N, b_layout));
    ck_tile::HostTensor<::EDataType> e_m_n_dev_result(
        f_host_tensor_descriptor(total_G, M, N, stride_E, M * N, e_layout));

    ck_tile::FillUniformDistribution<::ADataType>{-5.f, 5.f}(a_m_k);
    ck_tile::FillUniformDistribution<::BDataType>{-5.f, 5.f}(b_k_n);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem e_m_n_dev_buf(e_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    e_m_n_dev_buf.SetZero();
    e_m_n_dev_result.SetZero();

    using DsDataType = ck_tile::tuple<>;
    using DsLayout   = ck_tile::tuple<>;

    float ave_time = invoke_batched_contraction<::ADataType,
                                                ::BDataType,
                                                DsDataType,
                                                ::AccDataType,
                                                ::EDataType,
                                                ALayout,
                                                BLayout,
                                                DsLayout,
                                                ELayout>(a_m_k_dev_buf,
                                                         b_k_n_dev_buf,
                                                         e_m_n_dev_buf,
                                                         M,
                                                         N,
                                                         K,
                                                         G_dims,
                                                         stride_A,
                                                         stride_B,
                                                         stride_E,
                                                         G_strides_A,
                                                         G_strides_B,
                                                         G_strides_E,
                                                         kbatch,
                                                         n_warmup,
                                                         n_repeat);

    e_m_n_dev_buf.FromDevice(e_m_n_dev_result.data());

    // Print G dimensions info
    std::string g_info = "G_dims=[";
    for(size_t i = 0; i < G_dims.size(); ++i)
    {
        g_info += std::to_string(G_dims[i]);
        if(i < G_dims.size() - 1)
            g_info += ",";
    }
    g_info += "] (total=" + std::to_string(total_G) + ")";

    std::string op_name{"Batched Contraction : G: " + std::to_string(G_dims.size()) + "D"};
    std::size_t flop     = std::size_t(2) * total_G * M * N * K;
    std::size_t num_byte = sizeof(::ADataType) * total_G * M * K +
                           sizeof(::BDataType) * total_G * N * K +
                           sizeof(::EDataType) * total_G * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run " << op_name << " kernel with M=" << M << " N=" << N << " K=" << K << " "
              << g_info << " : " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
              << " GB/s" << std::endl;

    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<::EDataType> e_m_n_host_ref(
            f_host_tensor_descriptor(total_G, M, N, stride_E, M * N, ELayout{}));
        e_m_n_host_ref.SetZero();

        // Simple matrix multiplication reference
        for(ck_tile::index_t flat_g = 0; flat_g < total_G; ++flat_g)
        {
            for(ck_tile::index_t m = 0; m < M; ++m)
            {
                for(ck_tile::index_t n = 0; n < N; ++n)
                {
                    ::AccDataType sum = 0;
                    for(ck_tile::index_t k = 0; k < K; ++k)
                    {
                        sum += static_cast<::AccDataType>(a_m_k(flat_g, m, k)) *
                               static_cast<::AccDataType>(b_k_n(flat_g, k, n));
                    }
                    e_m_n_host_ref(flat_g, m, n) = static_cast<::EDataType>(sum);
                }
            }
        }

        const float max_accumulated_value =
            *std::max_element(e_m_n_host_ref.mData.begin(), e_m_n_host_ref.mData.end());

        const auto rtol_atol =
            calculate_rtol_atol<::ADataType, ::BDataType, ::EDataType, ::AccDataType>(
                K, kbatch, max_accumulated_value);

        pass = ck_tile::check_err(e_m_n_dev_result,
                                  e_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;

        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

int run_batched_contraction_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "C")
    {
        return run_batched_contraction_example_with_layouts(argc, argv, Row{}, Col{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and E tensors! "
                                 "Only R-C-R supported for now.");
    }
}
