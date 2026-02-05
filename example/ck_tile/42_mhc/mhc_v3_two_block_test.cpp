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
    // TESTING: Use 2 blocks worth of batch
    const int B          = 128;           // Two blocks worth of batch
    const int n          = 4;             // Expansion rate (aka streams)
    const int C          = 4096;          // Output layer dim
    const int nC         = n * C;         // Total input dimension
    const int output_dim = 2 * n + n * n; // 2n + n^2 = 24

    using ActivationFunc = ck_tile::element_wise::Sigmoid;

    std::cout << "\n--- Testing MHC Kernel V3 TWO BLOCKS with B=" << B << " (n=" << n << ", C=" << C
              << ") ---" << std::endl;
    std::cout << "Output dimension: " << output_dim << std::endl;

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

    // V3 kernel with 2D tiling
    constexpr ck_tile::index_t kMTile = 64; // Batch tile
    constexpr ck_tile::index_t kNTile = 32; // Output tile
    constexpr ck_tile::index_t kKTile = 8;  // K tile

    using KernelV3 = ck_tile::
        MHCKernelV3<Problem, ck_tile::MHCDefaultPolicy, kMTile, kNTile, kKTile, ActivationFunc>;

    const ck_tile::index_t kBlockSize = KernelV3::BlockSize();

    // Should be exactly 2 blocks
    auto grid_size = KernelV3::GetGridSize(B, output_dim);
    const ck_tile::index_t kGridSize =
        grid_size.at(ck_tile::number<0>{}) * grid_size.at(ck_tile::number<1>{});

    std::cout << "Grid configuration: " << grid_size.at(ck_tile::number<0>{}) << " × "
              << grid_size.at(ck_tile::number<1>{}) << " = " << kGridSize << " blocks (SHOULD BE 2)"
              << std::endl;
    std::cout << "Block size: " << kBlockSize << " threads" << std::endl;
    std::cout << "Shared memory: " << KernelV3::GetSmemSize() << " bytes" << std::endl;

    constexpr ck_tile::index_t kBlockPerCu = 1;

    const float r = 2.0f, alpha_pre = 1.5f, alpha_post = 2.5f, alpha_res = 3.5f, bias = 1.5f;

    // Launch kernel
    ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, false, 0},
        ck_tile::make_kernel<kBlockPerCu>(KernelV3{},
                                          kGridSize,
                                          kBlockSize,
                                          KernelV3::GetSmemSize(),
                                          static_cast<float*>(d_x_mem.GetDeviceBuffer()),
                                          static_cast<float*>(d_phi_mem.GetDeviceBuffer()),
                                          static_cast<float*>(d_output_mem.GetDeviceBuffer()),
                                          B,
                                          nC,
                                          output_dim,
                                          n,
                                          r,
                                          alpha_pre,
                                          alpha_post,
                                          alpha_res,
                                          bias));

    d_output_mem.FromDevice(h_output.data());

    // Compute reference
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
        h_output, h_output_ref, "Error: MHC V3 TWO BLOCKS kernel output mismatch!", 1e-3f, 1e-3f);

    std::cout << "Result: " << (pass ? "PASS" : "FAIL") << std::endl;

    if(!pass)
    {
        // Print first and second batch values for debugging
        std::cout << "\nFirst batch (0) kernel output: [";
        for(int i = 0; i < std::min(8, output_dim); i++)
        {
            std::cout << h_output(0, i);
            if(i < std::min(8, output_dim) - 1)
                std::cout << ", ";
        }
        std::cout << " ...]" << std::endl;

        std::cout << "First batch (0) reference: [";
        for(int i = 0; i < std::min(8, output_dim); i++)
        {
            std::cout << h_output_ref(0, i);
            if(i < std::min(8, output_dim) - 1)
                std::cout << ", ";
        }
        std::cout << " ...]" << std::endl;

        std::cout << "\nSecond block first batch (64) kernel output: [";
        for(int i = 0; i < std::min(8, output_dim); i++)
        {
            std::cout << h_output(64, i);
            if(i < std::min(8, output_dim) - 1)
                std::cout << ", ";
        }
        std::cout << " ...]" << std::endl;

        std::cout << "Second block first batch (64) reference: [";
        for(int i = 0; i < std::min(8, output_dim); i++)
        {
            std::cout << h_output_ref(64, i);
            if(i < std::min(8, output_dim) - 1)
                std::cout << ", ";
        }
        std::cout << " ...]" << std::endl;
    }

    return pass ? 0 : 1;
}
