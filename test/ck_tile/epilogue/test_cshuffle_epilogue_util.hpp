// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>

namespace ck_tile {

    // CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    // {
    //     constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
    //         sequence<>,
    //         tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
    //         tuple<sequence<1, 2>>,
    //         tuple<sequence<1, 1>>,
    //         sequence<1, 2>,
    //         sequence<0, 0>>{};

    //     constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
    //         c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
    //     constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
    //     auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);

    //     return c_block_tensor;
    // }
// Simple test kernel to invoke the CShuffleEpilogue
template <typename Problem, index_t M, index_t N>
__global__ void
test_cshuffle_epilogue_kernel(typename Problem::ODataType* __restrict__ output_data)
{
    using Epilogue = CShuffleEpilogue<Problem>;
    
    static_assert(Problem::kMPerBlock <= M && Problem::kNPerBlock <= N, 
                  "Block size must fit in tensor dimensions");
    
    // Allocate shared memory for epilogue
    __shared__ char smem[Epilogue::GetSmemSize()];
    
    // Create accumulator tile
    constexpr auto lds_distribution_encode = make_static_tile_distribution(Epilogue::MakeLdsDistributionEncode());
    auto acc_tile = make_static_distributed_tensor<float/*typename Epilogue::ODataType*/>(lds_distribution_encode);

    // Fill acc_tile with a simple pattern
    auto& acc_buffer = acc_tile.get_thread_buffer();
    acc_buffer[0] = 2.0F;
    
    // Create output tensor view
    auto output_tensor_view = make_naive_tensor_view<address_space_enum::global>(
        output_data,
        make_tuple(M, N),
        make_tuple(N, 1),
        number<Epilogue::GetVectorSizeC()>{},
        number<1>{});
    
    // Create output tile window
    auto output_tile_window = make_tile_window(
        output_tensor_view,
        make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}),
        {0, 0});
    
    // Create empty D tensors tuple (we're ignoring ds_dram_windows for this test)
    auto empty_ds = make_tuple();
    
    // Call the epilogue
    Epilogue{}(output_tile_window, acc_tile, empty_ds, smem);
}

// Test configuration helper
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename ODataType,
          index_t kBlockSize,
          index_t kM,
          index_t kN,
          index_t MWave,
          index_t NWave,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t KPerXdl,
          bool isCTransposed = false>
using SimpleCShuffleEpilogueProblem = CShuffleEpilogueProblem<
    ADataType,
    BDataType,
    ck_tile::tuple<>, // Empty Ds tuple
    AccDataType,
    ODataType,
    ck_tile::tuple<>, // Empty Ds layout
    tensor_layout::gemm::RowMajor, // ELayout
    ck_tile::element_wise::PassThrough,     // CDElementwise
    kBlockSize,
    kM,
    kN,
    MWave,
    NWave,
    MPerXdl,
    NPerXdl,
    KPerXdl,
    isCTransposed,
    memory_operation_enum::set>;

template <typename Problem, index_t M, index_t N>
bool run_cshuffle_epilogue_test()
{
    using ODataType = typename Problem::ODataType;
    
    constexpr index_t kMPerBlock = Problem::kMPerBlock;
    constexpr index_t kNPerBlock = Problem::kNPerBlock;
    constexpr index_t kBlockSize = Problem::kBlockSize;
    
    std::cout << "Running CShuffleEpilogue test with M=" << M << ", N=" << N 
              << ", MPerBlock=" << kMPerBlock << ", NPerBlock=" << kNPerBlock 
              << ", BlockSize=" << kBlockSize << std::endl;
    
    // Allocate host memory
    const size_t output_size = M * N;
    
    std::vector<ODataType> host_output(output_size, static_cast<ODataType>(0));
    
    // Allocate device memory
    ODataType* device_output;
    
    auto hip_err = hipMalloc(&device_output, output_size * sizeof(ODataType));
    if(hip_err != hipSuccess) {
        std::cerr << "hipMalloc failed: " << hipGetErrorString(hip_err) << std::endl;
        return false;
    }
    
    hip_err = hipMemcpy(device_output, host_output.data(), output_size * sizeof(ODataType), hipMemcpyHostToDevice);
    if(hip_err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(hip_err) << std::endl;
        hip_err = hipFree(device_output);
        (void)hip_err; // Suppress unused variable warning
        return false;
    }
    
    // Launch kernel
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(kBlockSize, 1, 1);
    
    test_cshuffle_epilogue_kernel<Problem, M, N><<<gridSize, blockSize>>>(
        device_output);
    
    // Check for kernel launch errors
    auto hipError = hipGetLastError();
    if (hipError != hipSuccess) {
        std::cout << "Kernel launch failed: " << hipGetErrorString(hipError) << std::endl;
        return false;
    }
    
    hip_err = hipDeviceSynchronize();
    if(hip_err != hipSuccess) {
        std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(hip_err) << std::endl;
        auto free_err = hipFree(device_output);
        (void)free_err; // Suppress unused variable warning
        return false;
    }
    
    // Check for kernel execution errors
    hipError = hipGetLastError();
    if (hipError != hipSuccess) {
        std::cout << "Kernel execution failed: " << hipGetErrorString(hipError) << std::endl;
        auto free_err = hipFree(device_output);
        (void)free_err; // Suppress unused variable warning
        return false;
    }
    
    // Copy results back
    hip_err = hipMemcpy(host_output.data(), device_output, output_size * sizeof(ODataType), hipMemcpyDeviceToHost);
    if(hip_err != hipSuccess) {
        std::cerr << "hipMemcpy D2H failed: " << hipGetErrorString(hip_err) << std::endl;
        auto free_err = hipFree(device_output);
        (void)free_err; // Suppress unused variable warning
        return false;
    }
    
    // Basic verification - just check that output has a 2
    bool has_2 = false;
    for (size_t i = 0; i < output_size; ++i) {
        if (host_output[i] > static_cast<ODataType>(1.9F) && host_output[i] < static_cast<ODataType>(2.1F)) {
            has_2 = true;
            break;
        }
    }
    
    // Cleanup
    auto free_err = hipFree(device_output);
    (void)free_err; // Suppress unused variable warning
    
    return has_2; // Return true if we found any non-zero output
}

} // namespace ck_tile
