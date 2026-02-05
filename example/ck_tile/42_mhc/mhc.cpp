// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/mhc.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_mhc.hpp"
#include "ck_tile/host/check_err.hpp"

int main()
{

    const int B          = 1024;          // Batch size
    const int n          = 4;             // Expansion rate (aka streams)
    const int C          = 4096;          // Output layer dim
    const int nC         = n * C;         // Total input dimension
    const int output_dim = 2 * n + n * n; // 2n + n^2

    using ActivationFunc = ck_tile::element_wise::Sigmoid; // Example activation function

    std::cout << "\n--- Testing batch size B=" << B << " (n=" << n << ", C=" << C
              << ") with activation: " << ActivationFunc::name << " ---" << std::endl;

    // Allocate host tensors
    ck_tile::HostTensor<float> h_x({B, nC});
    ck_tile::HostTensor<float> h_phi({nC, output_dim});
    ck_tile::HostTensor<float> h_output({B, output_dim});

    // Initialize with random data
    ck_tile::FillUniformDistribution<float>{-1.0f, 1.0f}(h_x);
    ck_tile::FillUniformDistribution<float>{-0.5f, 0.5f}(h_phi);
    h_output.SetZero();

    // Allocate device memory
    ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_phi_mem(h_phi.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_output_mem(h_output.get_element_space_size_in_bytes());

    // Copy data to device
    d_x_mem.ToDevice(h_x.data());
    d_phi_mem.ToDevice(h_phi.data());
    d_output_mem.ToDevice(h_output.data());

    // Define block shape
    using BlockShape = ck_tile::Generic2dBlockShape<ck_tile::sequence<1, 256>,
                                                    ck_tile::sequence<1, 256>,
                                                    ck_tile::sequence<1, 1>>;

    using Problem = ck_tile::MHCProblem<float, float, float, BlockShape>;

    // Use template parameters for B, n, C, and Activation (compile-time)
    using KernelExpansionParallel =
        ck_tile::ManifoldConstrainedHyperConnectionTiled<Problem,
                                                         ck_tile::MHCDefaultPolicy,
                                                         B,
                                                         n,
                                                         C,
                                                         256,
                                                         ActivationFunc>;

    const ck_tile::index_t kBlockSize      = KernelExpansionParallel::BlockSize();
    const ck_tile::index_t kGridSize       = (output_dim + 15) / 16;
    constexpr ck_tile::index_t kBlockPerCu = 1;

    const float r = 2.0f, alpha_pre = 1.5f, alpha_post = 2.5f, alpha_res = 3.5f, bias = 1.5f;

    // Launch kernel
    ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, false, 0},
        ck_tile::make_kernel<kBlockPerCu>(KernelExpansionParallel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          static_cast<float*>(d_x_mem.GetDeviceBuffer()),
                                          static_cast<float*>(d_phi_mem.GetDeviceBuffer()),
                                          static_cast<float*>(d_output_mem.GetDeviceBuffer()),
                                          r,
                                          alpha_pre,
                                          alpha_post,
                                          alpha_res,
                                          bias));

    d_output_mem.FromDevice(h_output.data());

    // Compute reference with the same activation function
    ck_tile::HostTensor<float> h_output_ref({B, output_dim});
    h_output_ref.SetZero();
    ck_tile::reference_mhc<float, float, float, float, ActivationFunc>(h_x,
                                                                       h_phi,
                                                                       h_output_ref,
                                                                       n,
                                                                       C,
                                                                       r,
                                                                       alpha_pre,
                                                                       alpha_post,
                                                                       alpha_res,
                                                                       bias,
                                                                       ActivationFunc{});

    // Validate
    bool pass = ck_tile::check_err(
        h_output, h_output_ref, "Error: Activation function mismatch!", 1e-3f, 1e-3f);

    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;

    if(!pass)
    {
        // Print first few values for debugging
        std::cout << "  First batch kernel output: [";
        for(int i = 0; i < std::min(8, output_dim); i++)
        {
            std::cout << h_output(0, i);
            if(i < std::min(8, output_dim) - 1)
                std::cout << ", ";
        }
        std::cout << " ...]" << std::endl;

        std::cout << "  First batch reference: [";
        for(int i = 0; i < std::min(8, output_dim); i++)
        {
            std::cout << h_output_ref(0, i);
            if(i < std::min(8, output_dim) - 1)
                std::cout << ", ";
        }
        std::cout << " ...]" << std::endl;
    }
}
