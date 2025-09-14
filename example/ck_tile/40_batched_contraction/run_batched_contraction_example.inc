// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include "contraction_utils.hpp"

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

    e_full_dims_dev_buf.FromDevice(e_full_dims_dev.data());

    // This function loops over ANY number of dimensions
    std::function<void(std::vector<std::size_t>&,
                       const std::vector<ck_tile::index_t>&,
                       std::size_t,
                       std::function<void()>)>
        loop_dimensions =
            [&](std::vector<std::size_t>& indices, // Current index values [i0, i1, i2, ...]
                const std::vector<ck_tile::index_t>&
                    dims,                           // Dimension sizes [dim0, dim1, dim2, ...]
                std::size_t dim_idx,                // Which dimension we're currently looping
                std::function<void()> inner_func) { // What to do when all dimensions are set
                if(dim_idx >= dims.size())
                {
                    inner_func(); // All dimensions set, execute the inner computation
                    return;
                }

                // Loop over current dimension
                for(ck_tile::index_t i = 0; i < dims[dim_idx]; ++i)
                {
                    indices[dim_idx] = i; // Set current dimension index
                    loop_dimensions(
                        indices, dims, dim_idx + 1, inner_func); // Recurse to next dimension
                }
            };

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<::EDataType> e_full_dims_host_ref(
            ck_tile::HostTensorDescriptor(E_dims, E_strides));
        e_full_dims_host_ref.SetZero();

        std::cout << "Computing CPU reference..." << std::endl;

        // progress tracking
        auto start_time                        = std::chrono::high_resolution_clock::now();
        ck_tile::index_t total_output_elements = G_total * M_total * N_total;
        ck_tile::index_t completed_elements    = 0;
        ck_tile::index_t next_progress_report  = total_output_elements / 10; // Report every 10%

        std::vector<std::size_t> g_indices(G_dims.size());
        std::vector<std::size_t> m_indices(M_dims.size());
        std::vector<std::size_t> n_indices(N_dims.size());
        std::vector<std::size_t> k_indices(K_dims.size());

        loop_dimensions(g_indices, G_dims, 0, [&]() {
            loop_dimensions(m_indices, M_dims, 0, [&]() {
                loop_dimensions(n_indices, N_dims, 0, [&]() {
                    ::AccDataType sum = 0;

                    loop_dimensions(k_indices, K_dims, 0, [&]() {
                        std::vector<std::size_t> a_idx, b_idx;

                        // A[G_indices, M_indices, K_indices]
                        a_idx.insert(a_idx.end(), g_indices.begin(), g_indices.end());
                        a_idx.insert(a_idx.end(), m_indices.begin(), m_indices.end());
                        a_idx.insert(a_idx.end(), k_indices.begin(), k_indices.end());

                        // B[G_indices, N_indices, K_indices]
                        b_idx.insert(b_idx.end(), g_indices.begin(), g_indices.end());
                        b_idx.insert(b_idx.end(), n_indices.begin(), n_indices.end());
                        b_idx.insert(b_idx.end(), k_indices.begin(), k_indices.end());

                        auto a_val = a_full_dims(a_idx);
                        auto b_val = b_full_dims(b_idx);

                        sum +=
                            static_cast<::AccDataType>(a_val) * static_cast<::AccDataType>(b_val);
                    });

                    std::vector<std::size_t> e_idx;
                    e_idx.insert(e_idx.end(), g_indices.begin(), g_indices.end());
                    e_idx.insert(e_idx.end(), m_indices.begin(), m_indices.end());
                    e_idx.insert(e_idx.end(), n_indices.begin(), n_indices.end());

                    e_full_dims_host_ref(e_idx) = static_cast<::EDataType>(sum);

                    ++completed_elements;
                    if(completed_elements >= next_progress_report ||
                       completed_elements == total_output_elements)
                    {
                        auto now = std::chrono::high_resolution_clock::now();
                        auto elapsed =
                            std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);

                        int percentage = (100 * completed_elements) / total_output_elements;
                        std::cout << "Progress: " << percentage << "% (" << completed_elements
                                  << "/" << total_output_elements << " elements, "
                                  << elapsed.count() << "ms)" << std::endl;

                        next_progress_report += total_output_elements / 10; // Next 10%
                    }
                });
            });
        });
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "CPU reference completed in " << duration.count() << "ms" << std::endl;
    }
    bool pass = true;
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
