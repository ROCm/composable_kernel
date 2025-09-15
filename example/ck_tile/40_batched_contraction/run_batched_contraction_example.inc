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

void calculate_reference_multi_dimensional(const ck_tile::HostTensor<::ADataType>& a_full_dims,
                                           const ck_tile::HostTensor<::BDataType>& b_full_dims,
                                           ck_tile::HostTensor<::EDataType>& e_full_dims_host_ref,
                                           const std::vector<ck_tile::index_t>& G_dims,
                                           const std::vector<ck_tile::index_t>& M_dims,
                                           const std::vector<ck_tile::index_t>& N_dims,
                                           const std::vector<ck_tile::index_t>& K_dims,
                                           const std::vector<ck_tile::index_t>& A_dims,
                                           const std::vector<ck_tile::index_t>& B_dims,
                                           const std::vector<ck_tile::index_t>& E_dims)
{
    std::cout << "Calculating reference using multi-dimensional indexing..." << std::endl;

    std::vector<std::size_t> g_idx(G_dims.size());
    std::vector<std::size_t> m_idx(M_dims.size());
    std::vector<std::size_t> n_idx(N_dims.size());
    std::vector<std::size_t> k_idx(K_dims.size());
    std::vector<std::size_t> a_idx, b_idx, e_idx;

    a_idx.reserve(A_dims.size());
    b_idx.reserve(B_dims.size());
    e_idx.reserve(E_dims.size());

    for(ck_tile::index_t g_flat = 0; g_flat < calculate_total_elements(G_dims); ++g_flat)
    {
        ck_tile::index_t temp = g_flat;
        for(int i = G_dims.size() - 1; i >= 0; --i)
        {
            g_idx[i] = temp % G_dims[i];
            temp /= G_dims[i];
        }

        for(ck_tile::index_t m_flat = 0; m_flat < calculate_total_elements(M_dims); ++m_flat)
        {
            temp = m_flat;
            for(int i = M_dims.size() - 1; i >= 0; --i)
            {
                m_idx[i] = temp % M_dims[i];
                temp /= M_dims[i];
            }

            for(ck_tile::index_t n_flat = 0; n_flat < calculate_total_elements(N_dims); ++n_flat)
            {
                temp = n_flat;
                for(int i = N_dims.size() - 1; i >= 0; --i)
                {
                    n_idx[i] = temp % N_dims[i];
                    temp /= N_dims[i];
                }

                ::AccDataType sum = 0;

                for(ck_tile::index_t k_flat = 0; k_flat < calculate_total_elements(K_dims);
                    ++k_flat)
                {
                    temp = k_flat;
                    for(int i = K_dims.size() - 1; i >= 0; --i)
                    {
                        k_idx[i] = temp % K_dims[i];
                        temp /= K_dims[i];
                    }

                    a_idx.clear();
                    b_idx.clear();

                    a_idx.insert(a_idx.end(), g_idx.begin(), g_idx.end());
                    a_idx.insert(a_idx.end(), m_idx.begin(), m_idx.end());
                    a_idx.insert(a_idx.end(), k_idx.begin(), k_idx.end());

                    b_idx.insert(b_idx.end(), g_idx.begin(), g_idx.end());
                    b_idx.insert(b_idx.end(), n_idx.begin(), n_idx.end());
                    b_idx.insert(b_idx.end(), k_idx.begin(), k_idx.end());

                    auto a_val = a_full_dims(a_idx);
                    auto b_val = b_full_dims(b_idx);

                    sum += static_cast<::AccDataType>(a_val) * static_cast<::AccDataType>(b_val);
                }

                e_idx.clear();
                e_idx.insert(e_idx.end(), g_idx.begin(), g_idx.end());
                e_idx.insert(e_idx.end(), m_idx.begin(), m_idx.end());
                e_idx.insert(e_idx.end(), n_idx.begin(), n_idx.end());

                e_full_dims_host_ref(e_idx) = static_cast<::EDataType>(sum);
            }
        }
    }
}

void calculate_reference_flat_indexing(const ck_tile::HostTensor<::ADataType>& a_full_dims,
                                       const ck_tile::HostTensor<::BDataType>& b_full_dims,
                                       ck_tile::HostTensor<::EDataType>& e_full_dims_host_ref,
                                       ck_tile::index_t G_total,
                                       ck_tile::index_t M_total,
                                       ck_tile::index_t N_total,
                                       ck_tile::index_t K_total)
{
    std::cout << "Calculating reference using flat indexing..." << std::endl;

    for(ck_tile::index_t g_flat = 0; g_flat < G_total; ++g_flat)
    {
        for(ck_tile::index_t m_flat = 0; m_flat < M_total; ++m_flat)
        {
            for(ck_tile::index_t n_flat = 0; n_flat < N_total; ++n_flat)
            {
                ::AccDataType sum = 0;
                for(ck_tile::index_t k_flat = 0; k_flat < K_total; ++k_flat)
                {
                    auto a_val =
                        a_full_dims.mData[g_flat * M_total * K_total + m_flat * K_total + k_flat];
                    auto b_val =
                        b_full_dims.mData[g_flat * N_total * K_total + n_flat * K_total + k_flat];
                    sum += static_cast<::AccDataType>(a_val) * static_cast<::AccDataType>(b_val);
                }
                e_full_dims_host_ref.mData[g_flat * M_total * N_total + m_flat * N_total + n_flat] =
                    static_cast<::EDataType>(sum);
            }
        }
    }
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
float invoke_batched_contraction_kernel(
    ck_tile::DeviceMem& a_full_dims_dev_buf,
    ck_tile::DeviceMem& b_full_dims_dev_buf,
    ck_tile::DeviceMem& e_full_dims_dev_buf,
    const std::vector<ck_tile::index_t>& G_dims,
    const std::vector<ck_tile::index_t>& M_dims,
    const std::vector<ck_tile::index_t>& N_dims,
    const std::vector<ck_tile::index_t>& K_dims,
    const std::vector<ck_tile::index_t>& A_dims,    // [G0,G1,..,M0,M1,..,K0,K1,..]
    const std::vector<ck_tile::index_t>& B_dims,    // [G0,G1,..,N0,N1,..,K0,K1,..]
    const std::vector<ck_tile::index_t>& E_dims,    // [G0,G1,..,M0,M1,..,N0,N1,..]
    const std::vector<ck_tile::index_t>& A_strides, // [G0,G1,..,M0,M1,..,K0,K1,..]
    const std::vector<ck_tile::index_t>& B_strides, // [G0,G1,..,N0,N1,..,K0,K1,..]
    const std::vector<ck_tile::index_t>& E_strides, // [G0,G1,..,M0,M1,..,N0,N1,..]
    ck_tile::index_t G_total,
    ck_tile::index_t M_total,
    ck_tile::index_t N_total,
    ck_tile::index_t K_total,
    ck_tile::index_t kbatch,
    int n_warmup,
    int n_repeat)
{
    std::cout << "Creating BatchedContractionHostArgs..." << std::endl;
    constexpr ck_tile::index_t NumDTensor = 0;
    std::array<const void*, NumDTensor> empty_ds_ptr{};
    std::array<std::vector<ck_tile::index_t>, NumDTensor> empty_ds_dims{};
    std::array<std::vector<ck_tile::index_t>, NumDTensor> empty_ds_strides{};

    ck_tile::BatchedContractionHostArgs<NumDTensor> args(
        a_full_dims_dev_buf.GetDeviceBuffer(), // a_ptr
        b_full_dims_dev_buf.GetDeviceBuffer(), // b_ptr
        empty_ds_ptr,                          // ds_ptr
        e_full_dims_dev_buf.GetDeviceBuffer(), // e_ptr
        kbatch,                                // k_batch
        G_total,                               // G_total
        M_total,                               // M_total
        N_total,                               // N_total
        K_total,                               // K_total
        A_dims,                                // A_dims
        B_dims,                                // B_dims
        empty_ds_dims,                         // Ds_dims
        E_dims,                                // E_dims
        A_strides,                             // A_strides
        B_strides,                             // B_strides
        empty_ds_strides,                      // Ds_strides
        E_strides                              // E_strides
    );

    std::cout << "Calling batched_contraction with dimensions: G=" << G_dims.size()
              << ", M=" << M_dims.size() << ", N=" << N_dims.size() << ", K=" << K_dims.size()
              << std::endl;

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
        args,
        ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
        G_dims.size(), // num_g_dims
        M_dims.size(), // num_m_dims
        N_dims.size(), // num_n_dims
        K_dims.size()  // num_k_dims
    );

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
    (void)a_layout;
    (void)b_layout;
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::vector<ck_tile::index_t> G_dims = parse_dimensions(arg_parser.get_str("g_dims"));
    std::vector<ck_tile::index_t> M_dims = parse_dimensions(arg_parser.get_str("m_dims"));
    std::vector<ck_tile::index_t> N_dims = parse_dimensions(arg_parser.get_str("n_dims"));
    std::vector<ck_tile::index_t> K_dims = parse_dimensions(arg_parser.get_str("k_dims"));

    ck_tile::index_t G_total = calculate_total_elements(G_dims);
    ck_tile::index_t M_total = calculate_total_elements(M_dims);
    ck_tile::index_t N_total = calculate_total_elements(N_dims);
    ck_tile::index_t K_total = calculate_total_elements(K_dims);

    std::vector<ck_tile::index_t> A_dims =
        get_tensor_dims({G_dims, M_dims, K_dims}); // [G0,G1,..,M0,M1,..,K0,K1,..]
    std::vector<ck_tile::index_t> B_dims =
        get_tensor_dims({G_dims, N_dims, K_dims}); // [G0,G1,..,N0,N1,..,K0,K1,..]
    std::vector<ck_tile::index_t> E_dims =
        get_tensor_dims({G_dims, M_dims, N_dims}); // [G0,G1,..,M0,M1,..,N0,N1,..]

    std::vector<ck_tile::index_t> A_strides = get_tensor_strides(A_dims);
    std::vector<ck_tile::index_t> B_strides = get_tensor_strides(B_dims);
    std::vector<ck_tile::index_t> E_strides = get_tensor_strides(E_dims);

    ck_tile::index_t kbatch = arg_parser.get_int("split_k");
    (void)kbatch;
    int n_warmup = arg_parser.get_int("warmup");
    int n_repeat = arg_parser.get_int("repeat");
    (void)n_warmup;
    (void)n_repeat;

    print_dims("G_dims", G_dims, G_total);
    print_dims("M_dims", M_dims, M_total);
    print_dims("N_dims", N_dims, N_total);
    print_dims("K_dims", K_dims, K_total);

    std::cout << "\n=== Tensor Shapes for Kernel ===" << std::endl;
    print_dims("A_dims", A_dims, 0);
    print_dims("B_dims", B_dims, 0);
    print_dims("E_dims", E_dims, 0);

    std::cout << "\n=== Tensor Strides ===" << std::endl;
    print_dims("A_strides", A_strides, 0);
    print_dims("B_strides", B_strides, 0);
    print_dims("E_strides", E_strides, 0);
    std::cout << "===============================================\n" << std::endl;

    ck_tile::HostTensor<::ADataType> a_full_dims(ck_tile::HostTensorDescriptor(A_dims, A_strides));
    ck_tile::HostTensor<::BDataType> b_full_dims(ck_tile::HostTensorDescriptor(B_dims, B_strides));
    ck_tile::HostTensor<::EDataType> e_full_dims_dev(
        ck_tile::HostTensorDescriptor(E_dims, E_strides));

    ck_tile::FillUniformDistribution<::ADataType>{-5.f, 5.f}(a_full_dims);
    ck_tile::FillUniformDistribution<::BDataType>{-5.f, 5.f}(b_full_dims);

    ck_tile::DeviceMem a_full_dims_dev_buf(a_full_dims.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_full_dims_dev_buf(b_full_dims.get_element_space_size_in_bytes());
    ck_tile::DeviceMem e_full_dims_dev_buf(e_full_dims_dev.get_element_space_size_in_bytes());

    a_full_dims_dev_buf.ToDevice(a_full_dims.data());
    b_full_dims_dev_buf.ToDevice(b_full_dims.data());
    e_full_dims_dev_buf.SetZero();
    e_full_dims_dev.SetZero();

    std::cout << "\n=== Running GPU Kernel ===" << std::endl;

    using DsDataType = ck_tile::tuple<>;
    using DsLayout   = ck_tile::tuple<>;

    float ave_time = invoke_batched_contraction_kernel<::ADataType,
                                                       ::BDataType,
                                                       DsDataType,
                                                       ::AccDataType,
                                                       ::EDataType,
                                                       ALayout,
                                                       BLayout,
                                                       DsLayout,
                                                       ELayout>(a_full_dims_dev_buf,
                                                                b_full_dims_dev_buf,
                                                                e_full_dims_dev_buf,
                                                                G_dims,
                                                                M_dims,
                                                                N_dims,
                                                                K_dims,
                                                                A_dims,
                                                                B_dims,
                                                                E_dims,
                                                                A_strides,
                                                                B_strides,
                                                                E_strides,
                                                                G_total,
                                                                M_total,
                                                                N_total,
                                                                K_total,
                                                                kbatch,
                                                                n_warmup,
                                                                n_repeat);

    std::string op_name{
        "Multi-Dimensional Batched Contraction : G: " + std::to_string(G_dims.size()) +
        "D, M: " + std::to_string(M_dims.size()) + "D, N: " + std::to_string(N_dims.size()) +
        "D, K: " + std::to_string(K_dims.size()) + "D"};

    std::size_t flop =
        std::size_t(2) * G_total * M_total * N_total * K_total; // Number of operations
    std::size_t num_byte = sizeof(::ADataType) * G_total * M_total * K_total + // A tensor size
                           sizeof(::BDataType) * G_total * N_total * K_total + // B tensor size
                           sizeof(::EDataType) * G_total * M_total * N_total;  // E tensor size

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time; // TFlops calculation
    float gb_per_sec = num_byte / 1.E6 / ave_time;                 //  GB/s calculation
    print_dims("G_dims", G_dims, G_total);
    print_dims("M_dims", M_dims, M_total);
    print_dims("N_dims", N_dims, N_total);
    print_dims("K_dims", K_dims, K_total);

    std::cout << "  Performance: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
              << " GB/s" << std::endl;

    // DETAILED: Tensor information
    std::cout << "\nDetailed Tensor Info:" << std::endl;
    std::cout << "  A tensor: " << G_total << " x " << M_total << " x " << K_total << " = "
              << G_total * M_total * K_total << " elements ("
              << (sizeof(::ADataType) * G_total * M_total * K_total) / 1024 / 1024 << " MB)"
              << std::endl;
    std::cout << "  B tensor: " << G_total << " x " << N_total << " x " << K_total << " = "
              << G_total * N_total * K_total << " elements ("
              << (sizeof(::BDataType) * G_total * N_total * K_total) / 1024 / 1024 << " MB)"
              << std::endl;
    std::cout << "  E tensor: " << G_total << " x " << M_total << " x " << N_total << " = "
              << G_total * M_total * N_total << " elements ("
              << (sizeof(::EDataType) * G_total * M_total * N_total) / 1024 / 1024 << " MB)"
              << std::endl;
    std::cout << "  Total memory: " << num_byte / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  Total FLOPs: " << flop / 1000000 << " million" << std::endl;

    std::cout << "===============================================" << std::endl;

    e_full_dims_dev_buf.FromDevice(e_full_dims_dev.data());
    std::cout << "GPU results retrieved from device." << std::endl;

    bool pass = true;
    if(arg_parser.get_int("v") == 1)
    {

        std::cout << "Computing CPU reference..." << std::endl;

        ck_tile::HostTensor<::EDataType> e_full_dims_host_ref(
            ck_tile::HostTensorDescriptor(E_dims, E_strides));
        e_full_dims_host_ref.SetZero();

        auto start_time = std::chrono::high_resolution_clock::now();

        calculate_reference_flat_indexing(
            a_full_dims, b_full_dims, e_full_dims_host_ref, G_total, M_total, N_total, K_total);

        ck_tile::HostTensor<::EDataType> e_full_dims_host_ref1(
            ck_tile::HostTensorDescriptor(E_dims, E_strides));
        e_full_dims_host_ref1.SetZero();
        calculate_reference_multi_dimensional(a_full_dims,
                                              b_full_dims,
                                              e_full_dims_host_ref1,
                                              G_dims,
                                              M_dims,
                                              N_dims,
                                              K_dims,
                                              A_dims,
                                              B_dims,
                                              E_dims);

        bool identical = true;
        for(std::size_t i = 0; i < e_full_dims_host_ref1.mData.size(); ++i)
        {
            if(std::fabs(e_full_dims_host_ref1.mData[i] - e_full_dims_host_ref.mData[i]) >
               std::numeric_limits<::EDataType>::epsilon() * 10)
            {
                identical = false;
                std::cout << "Difference at index " << i << ": "
                          << "ref1=" << static_cast<float>(e_full_dims_host_ref1.mData[i]) << ", "
                          << "ref=" << static_cast<float>(e_full_dims_host_ref.mData[i])
                          << std::endl;
                break;
            }
        }

        std::cout << "Comparison result: " << (identical ? "IDENTICAL" : "DIFFERENT") << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "CPU reference completed in " << duration.count() << "ms" << std::endl;

        const float max_accumulated_value =
            *std::max_element(e_full_dims_host_ref.mData.begin(), e_full_dims_host_ref.mData.end());

        const auto rtol_atol =
            calculate_rtol_atol<::ADataType, ::BDataType, ::EDataType, ::AccDataType>(
                K_total, kbatch, max_accumulated_value);

        pass = ck_tile::check_err(e_full_dims_dev,
                                  e_full_dims_host_ref,
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
