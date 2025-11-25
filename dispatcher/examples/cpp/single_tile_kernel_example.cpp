// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Single CK Tile Kernel Integration Example
 *
 * Demonstrates dispatcher with ONE real generated CK Tile kernel.
 * The kernel header is included via compiler flag: -include <header>
 *
 * This follows the tile_engine benchmark pattern.
 */

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// The generated kernel header is included via -include compiler flag
// It defines:
// - using ADataType = ck_tile::half_t;
// - using BDataType = ck_tile::half_t;
// - using CDataType = ck_tile::half_t;
// - using AccDataType = float;
// - using ALayout = ...;
// - using BLayout = ...;
// - using CLayout = ...;
// - constexpr const char* KERNEL_NAME = "...";
// - struct SelectedKernel { ... };

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;

// Helper to check HIP errors
#define HIP_CHECK(call)                                                         \
    do                                                                          \
    {                                                                           \
        hipError_t err = call;                                                  \
        if(err != hipSuccess)                                                   \
        {                                                                       \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(err) << std::endl;                   \
            exit(1);                                                            \
        }                                                                       \
    } while(0)

KernelKey create_kernel_key()
{
    KernelKey key;

    // Signature
    key.signature.dtype_a             = DataType::FP16;
    key.signature.dtype_b             = DataType::FP16;
    key.signature.dtype_c             = DataType::FP16;
    key.signature.dtype_acc           = DataType::FP32;
    key.signature.layout_a            = LayoutTag::RowMajor;
    key.signature.layout_b            = LayoutTag::ColMajor;
    key.signature.layout_c            = LayoutTag::RowMajor;
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = false;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = SelectedKernel::UseStructuredSparsity;

    // Algorithm - extract from SelectedKernel
    key.algorithm.tile_shape.m      = SelectedKernel::TileM;
    key.algorithm.tile_shape.n      = SelectedKernel::TileN;
    key.algorithm.tile_shape.k      = SelectedKernel::TileK;
    key.algorithm.wave_shape.m      = SelectedKernel::WarpPerBlock_M;
    key.algorithm.wave_shape.n      = SelectedKernel::WarpPerBlock_N;
    key.algorithm.wave_shape.k      = SelectedKernel::WarpPerBlock_K;
    key.algorithm.warp_tile_shape.m = SelectedKernel::WarpTileM;
    key.algorithm.warp_tile_shape.n = SelectedKernel::WarpTileN;
    key.algorithm.warp_tile_shape.k = SelectedKernel::WarpTileK;
    key.algorithm.pipeline          = Pipeline::CompV4;
    key.algorithm.scheduler         = Scheduler::Intrawave;
    key.algorithm.epilogue          = Epilogue::CShuffle;
    key.algorithm.block_size        = SelectedKernel::BlockSize;
    key.algorithm.double_buffer     = SelectedKernel::DoubleSmemBuffer;
    key.algorithm.persistent        = SelectedKernel::UsePersistentKernel;
    key.algorithm.preshuffle        = SelectedKernel::Preshuffle;
    key.algorithm.transpose_c       = SelectedKernel::TransposeC;
    key.algorithm.num_wave_groups   = SelectedKernel::NumWaveGroups;
    key.gfx_arch                    = "gfx942";

    return key;
}

int main(int argc, char** argv)
{
    std::cout << "======================================================================\n";
    std::cout << "CK Tile Dispatcher - Single Kernel Integration Example\n";
    std::cout << "======================================================================\n\n";

    // GPU info
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    if(device_count == 0)
    {
        std::cerr << "No HIP devices found!\n";
        return 1;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (" << prop.gcnArchName << ")\n\n";

    // Register the kernel
    std::cout << "Registering kernel: " << KERNEL_NAME << "\n";

    auto key = create_kernel_key();
    std::cout << "  Kernel ID: " << key.encode_identifier() << "\n";
    std::cout << "  Tile: " << SelectedKernel::TileM << "x" << SelectedKernel::TileN << "x"
              << SelectedKernel::TileK << "\n";
    std::cout << "  Wave: " << SelectedKernel::WarpPerBlock_M << "x"
              << SelectedKernel::WarpPerBlock_N << "x" << SelectedKernel::WarpPerBlock_K << "\n\n";

    auto kernel =
        create_generated_tile_kernel<SelectedKernel, ADataType, BDataType, CDataType, AccDataType>(
            key, std::string(KERNEL_NAME));

    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Registry::Priority::High);

    // Enable auto-export to JSON - exports on program exit
    Registry::instance().enable_auto_export("dispatcher_kernels.json", true, false);
    std::cout << "Auto-export enabled: dispatcher_kernels.json\n\n";

    // Create dispatcher
    Dispatcher dispatcher;

    // Test problem sizes to validate timing
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}};

    std::cout << "Testing problem sizes:\n";
    std::cout << "------------------------------------------------------------------------\n";

    for(const auto& [M, N, K] : test_sizes)
    {
        Problem problem(M, N, K);

        // Allocate GPU memory
        ADataType *a_dev, *b_dev;
        CDataType* c_dev;
        HIP_CHECK(hipMalloc(&a_dev, M * K * sizeof(ADataType)));
        HIP_CHECK(hipMalloc(&b_dev, K * N * sizeof(BDataType)));
        HIP_CHECK(hipMalloc(&c_dev, M * N * sizeof(CDataType)));

        // Initialize with random data
        std::vector<ADataType> a_host(M * K);
        std::vector<BDataType> b_host(K * N);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for(auto& val : a_host)
            val = ADataType(dis(gen));
        for(auto& val : b_host)
            val = BDataType(dis(gen));

        HIP_CHECK(
            hipMemcpy(a_dev, a_host.data(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(b_dev, b_host.data(), K * N * sizeof(BDataType), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(c_dev, 0, M * N * sizeof(CDataType)));

        // Execute via dispatcher
        float time_ms = dispatcher.run(a_dev, b_dev, c_dev, problem, nullptr);

        // Calculate performance
        float tflops = (2.0f * M * N * K) / (time_ms * 1e9);

        std::cout << "  " << M << "x" << N << "x" << K << ": " << time_ms << " ms | " << tflops
                  << " TFLOPS\n";

        // Cleanup
        HIP_CHECK(hipFree(a_dev));
        HIP_CHECK(hipFree(b_dev));
        HIP_CHECK(hipFree(c_dev));
    }

    std::cout << "\n======================================================================\n";
    std::cout << "OK REAL CK Tile kernel executed successfully via dispatcher!\n";
    std::cout << "======================================================================\n";

    return 0;
}
