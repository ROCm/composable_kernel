// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include "contraction_utils.hpp"
#include "ck_tile/host/reference/reference_batched_contraction.hpp"

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
float invoke_batched_contraction_kernel(
    const void* a_full_dims_dev_buf,
    const void* b_full_dims_dev_buf,
    const std::array<const void*, DsDataType::size()>& ds_dev_buf,
    void* e_full_dims_dev_buf,
    const std::vector<ck_tile::index_t>& G_dims,
    const std::vector<ck_tile::index_t>& M_dims,
    const std::vector<ck_tile::index_t>& N_dims,
    const std::vector<ck_tile::index_t>& K_dims,
    const std::vector<ck_tile::index_t>& A_dims, // [G0,G1,..,M0,M1,..,K0,K1,..]
    const std::vector<ck_tile::index_t>& B_dims, // [G0,G1,..,N0,N1,..,K0,K1,..]
    const std::array<std::vector<ck_tile::index_t>, DsDataType::size()>&
        Ds_dims, // [G0, G1, ..., M0, M1, ... , N0, N1, ...][NumDTensor]
    const std::vector<ck_tile::index_t>& E_dims,    // [G0,G1,..,M0,M1,..,N0,N1,..]
    const std::vector<ck_tile::index_t>& A_strides, // [G0,G1,..,M0,M1,..,K0,K1,..]
    const std::vector<ck_tile::index_t>& B_strides, // [G0,G1,..,N0,N1,..,K0,K1,..]
    const std::array<std::vector<ck_tile::index_t>, DsDataType::size()>& Ds_strides,
    const std::vector<ck_tile::index_t>& E_strides, // [G0,G1,..,M0,M1,..,N0,N1,..]
    ck_tile::index_t kbatch,
    int n_warmup,
    int n_repeat)
{
    std::cout << "Creating BatchedContractionHostArgs..." << std::endl;

    ck_tile::BatchedContractionHostArgs<DsDataType::size()> args(a_full_dims_dev_buf, // a_ptr
                                                                 b_full_dims_dev_buf, // b_ptr
                                                                 ds_dev_buf,          // ds_ptr
                                                                 e_full_dims_dev_buf, // e_ptr
                                                                 kbatch,              // k_batch
                                                                 A_dims,              // A_dims
                                                                 B_dims,              // B_dims
                                                                 Ds_dims,             // Ds_dims
                                                                 E_dims,              // E_dims
                                                                 A_strides,           // A_strides
                                                                 B_strides,           // B_strides
                                                                 Ds_strides,          // Ds_strides
                                                                 E_strides            // E_strides
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

template <typename ALayout, typename BLayout, typename DLayout, typename ELayout>
int run_batched_contraction_example_with_layouts(
    int argc,
    char* argv[],
    [[maybe_unused]] const ALayout a_layout = ALayout{},
    [[maybe_unused]] const BLayout b_layout = BLayout{},
    [[maybe_unused]] const DLayout d_layout = DLayout{},
    [[maybe_unused]] const ELayout e_layout = ELayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::vector<ck_tile::index_t> G_dims = parse_dimensions(arg_parser.get_str("g_dims"));
    std::vector<ck_tile::index_t> M_dims = parse_dimensions(arg_parser.get_str("m_dims"));
    std::vector<ck_tile::index_t> N_dims = parse_dimensions(arg_parser.get_str("n_dims"));
    std::vector<ck_tile::index_t> K_dims = parse_dimensions(arg_parser.get_str("k_dims"));

    constexpr ck_tile::index_t NumDTensor = 2;

    ck_tile::index_t G_total = calculate_total_elements(G_dims);
    ck_tile::index_t M_total = calculate_total_elements(M_dims);
    ck_tile::index_t N_total = calculate_total_elements(N_dims);
    ck_tile::index_t K_total = calculate_total_elements(K_dims);

    std::vector<ck_tile::index_t> A_dims =
        concatenate_dim_components({G_dims, M_dims, K_dims}); // [G0,G1,..,M0,M1,..,K0,K1,..]
    std::vector<ck_tile::index_t> B_dims =
        concatenate_dim_components({G_dims, N_dims, K_dims}); // [G0,G1,..,N0,N1,..,K0,K1,..]
    std::vector<ck_tile::index_t> E_dims =
        concatenate_dim_components({G_dims, M_dims, N_dims}); // [G0,G1,..,M0,M1,..,N0,N1,..]

    std::array<std::vector<ck_tile::index_t>, NumDTensor> Ds_dims;
    for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
    {
        Ds_dims[d] = E_dims;
    }

    auto convert_strides = [](const std::vector<std::size_t>& strides) {
        std::vector<ck_tile::index_t> converted(strides.size());
        std::copy(strides.begin(), strides.end(), converted.begin());
        return converted;
    };

    ck_tile::HostTensorDescriptor a_desc(A_dims);
    ck_tile::HostTensorDescriptor b_desc(B_dims);
    ck_tile::HostTensorDescriptor e_desc(E_dims);
    std::array<ck_tile::HostTensorDescriptor, NumDTensor> ds_descs;
    for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
    {
        ds_descs[d] = ck_tile::HostTensorDescriptor(Ds_dims[d], e_desc.get_strides());
    }

    std::vector<ck_tile::index_t> A_strides = convert_strides(a_desc.get_strides());
    std::vector<ck_tile::index_t> B_strides = convert_strides(b_desc.get_strides());
    std::vector<ck_tile::index_t> E_strides = convert_strides(e_desc.get_strides());

    std::array<std::vector<ck_tile::index_t>, NumDTensor> Ds_strides;
    for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
    {
        Ds_strides[d] = convert_strides(ds_descs[d].get_strides());
    }

    ck_tile::index_t kbatch = arg_parser.get_int("split_k");
    int n_warmup            = arg_parser.get_int("warmup");
    int n_repeat            = arg_parser.get_int("repeat");

    print_dims("G_dims", G_dims, G_total);
    print_dims("M_dims", M_dims, M_total);
    print_dims("N_dims", N_dims, N_total);
    print_dims("K_dims", K_dims, K_total);

    std::cout << "NumDTensor: " << NumDTensor << std::endl;
    std::cout << "\n=== Tensor Shapes for Kernel ===" << std::endl;
    print_dims("A_dims", A_dims, 0);
    print_dims("B_dims", B_dims, 0);
    print_dims("E_dims", E_dims, 0);
    for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
    {
        print_dims("Ds[" + std::to_string(d) + "]_dims", Ds_dims[d], 0);
    }

    std::cout << "\n=== Tensor Strides ===" << std::endl;
    print_dims("A_strides", A_strides, 0);
    print_dims("B_strides", B_strides, 0);
    print_dims("E_strides", E_strides, 0);
    for(ck_tile::index_t d = 0; d < NumDTensor; ++d)
    {
        print_dims("Ds[" + std::to_string(d) + "]_strides", Ds_strides[d], 0);
    }

    std::cout << "===============================================\n" << std::endl;

    ck_tile::HostTensor<::ADataType> a_full_dims_host(a_desc);
    ck_tile::HostTensor<::BDataType> b_full_dims_host(b_desc);
    ck_tile::HostTensor<::EDataType> e_full_dims_host(e_desc);

    std::vector<ck_tile::HostTensor<::DDataType>> ds_full_dims_host;
    for(int d = 0; d < NumDTensor; ++d)
    {
        ds_full_dims_host.emplace_back(ck_tile::HostTensor<::DDataType>(ds_descs[d]));
    }

    ck_tile::FillUniformDistribution<::ADataType>{-5.f, 5.f, std::nullopt}(a_full_dims_host);
    ck_tile::FillUniformDistribution<::BDataType>{-5.f, 5.f, std::nullopt}(b_full_dims_host);

    ck_tile::DeviceMem a_full_dims_dev_buf(a_full_dims_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_full_dims_dev_buf(b_full_dims_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem e_full_dims_dev_buf(e_full_dims_host.get_element_space_size_in_bytes());

    a_full_dims_dev_buf.ToDevice(a_full_dims_host.data());
    b_full_dims_dev_buf.ToDevice(b_full_dims_host.data());

    for(int d = 0; d < NumDTensor; ++d)
    {
        ck_tile::FillUniformDistribution<::DDataType>{-2.f, 2.f, std::nullopt}(
            ds_full_dims_host[d]);
    }

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> ds_full_dims_dev_buf;
    for(int d = 0; d < NumDTensor; ++d)
    {
        ds_full_dims_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            ds_full_dims_host[d].get_element_space_size_in_bytes()));
        ds_full_dims_dev_buf[d]->ToDevice(ds_full_dims_host[d].data());
    }
    std::array<const void*, NumDTensor> ds_ptr_buf;
    for(int d = 0; d < NumDTensor; ++d)
    {
        ds_ptr_buf[d] = ds_full_dims_dev_buf[d]->GetDeviceBuffer();
    }

    e_full_dims_dev_buf.SetZero();
    e_full_dims_host.SetZero();

    std::cout << "\n=== Running GPU Kernel ===" << std::endl;

    using DsDataType = ck_tile::tuple_array<::DDataType, NumDTensor>;
    using DsLayout   = ck_tile::tuple_array<DLayout, NumDTensor>;
    using CDEElementWise =
        std::conditional_t<NumDTensor == 0, ck_tile::element_wise::PassThrough, AddDs>;

    float ave_time =
        invoke_batched_contraction_kernel<::ADataType,
                                          ::BDataType,
                                          DsDataType,
                                          ::AccDataType,
                                          ::EDataType,
                                          ALayout,
                                          BLayout,
                                          DsLayout,
                                          ELayout,
                                          CDEElementWise>(a_full_dims_dev_buf.GetDeviceBuffer(),
                                                          b_full_dims_dev_buf.GetDeviceBuffer(),
                                                          ds_ptr_buf,
                                                          e_full_dims_dev_buf.GetDeviceBuffer(),
                                                          G_dims,
                                                          M_dims,
                                                          N_dims,
                                                          K_dims,
                                                          A_dims,
                                                          B_dims,
                                                          Ds_dims,
                                                          E_dims,
                                                          A_strides,
                                                          B_strides,
                                                          Ds_strides,
                                                          E_strides,
                                                          kbatch,
                                                          n_warmup,
                                                          n_repeat);

    std::string op_name{
        "Multi-Dimensional Batched Contraction : G: " + std::to_string(G_dims.size()) +
        "D, M: " + std::to_string(M_dims.size()) + "D, N: " + std::to_string(N_dims.size()) +
        "D, K: " + std::to_string(K_dims.size()) + "D"};

    std::size_t flop = std::size_t(2) * G_total * M_total * N_total * K_total +
                       NumDTensor * K_total * M_total * N_total; // Number of operations
    std::size_t num_byte =
        sizeof(::ADataType) * G_total * M_total * K_total +              // A tensor size
        sizeof(::BDataType) * G_total * N_total * K_total +              // B tensor size
        sizeof(::DDataType) * NumDTensor * G_total * M_total * N_total + // D tensors
        sizeof(::EDataType) * G_total * M_total * N_total;               // E tensor size

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time; // TFlops calculation
    float gb_per_sec = num_byte / 1.E6 / ave_time;                 //  GB/s calculation
    print_dims("G_dims", G_dims, G_total);
    print_dims("M_dims", M_dims, M_total);
    print_dims("N_dims", N_dims, N_total);
    print_dims("K_dims", K_dims, K_total);

    std::cout << "  Performance: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
              << " GB/s" << std::endl;

    std::cout << "===============================================" << std::endl;

    e_full_dims_dev_buf.FromDevice(e_full_dims_host.data());
    std::cout << "GPU results retrieved from device." << std::endl;

    bool pass = true;
    if(arg_parser.get_int("v") == 1)
    {

        std::cout << "Computing CPU reference..." << std::endl;

        ck_tile::HostTensor<::EDataType> e_full_dims_host_ref(
            ck_tile::HostTensorDescriptor(E_dims, E_strides));
        e_full_dims_host_ref.SetZero();

        auto start_time = std::chrono::high_resolution_clock::now();

        calculate_reference_flat_indexing<ADataType,
                                          BDataType,
                                          DDataType,
                                          EDataType,
                                          AccDataType,
                                          CDEElementWise>(a_full_dims_host,
                                                          b_full_dims_host,
                                                          ds_full_dims_host,
                                                          e_full_dims_host_ref,
                                                          G_total,
                                                          M_total,
                                                          N_total,
                                                          K_total,
                                                          CDEElementWise{});

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "CPU reference completed in " << duration.count() << "ms" << std::endl;

        const float max_accumulated_value =
            *std::max_element(e_full_dims_host_ref.mData.begin(), e_full_dims_host_ref.mData.end());

        const auto rtol_atol =
            calculate_rtol_atol<::ADataType, ::BDataType, ::EDataType, ::AccDataType>(
                K_total, kbatch, max_accumulated_value);

        pass = ck_tile::check_err(e_full_dims_host,
                                  e_full_dims_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail") << std::endl;

        std::cout << "===============================================" << std::endl;

        std::cout << "\n=== Random Samples of Reference and Result ===" << std::endl;

        // Generate 10 random indices
        std::vector<std::size_t> random_indices;
        std::size_t total_elements = e_full_dims_host_ref.mData.size();
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<std::size_t> dist(0, total_elements - 1);

        for(int i = 0; i < 10; ++i)
        {
            random_indices.push_back(dist(rng));
        }

        // Print the values at the random indices
        for(std::size_t idx : random_indices)
        {
            std::cout << "Index " << idx << ": "
                      << "ref=" << static_cast<float>(e_full_dims_host_ref.mData[idx]) << ", "
                      << "GPU=" << static_cast<float>(e_full_dims_host.mData[idx]) << std::endl;
        }

        std::cout << "===============================================" << std::endl;
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
        return run_batched_contraction_example_with_layouts(argc, argv, Row{}, Col{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and E tensors! "
                                 "Only R-C-R supported for now.");
    }
}
