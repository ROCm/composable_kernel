// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/epilogue_chainer.hpp"
#include "ck_tile/ops/epilogue/cshuffle_chained_epilogues.hpp"

#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>

namespace ck_tile {

// Simple test kernel to invoke the EpilogueChainer
template <typename Problem, index_t M, index_t N, bool UseScale>
__global__ void test_epilogue_chainer_kernel(typename Problem::ODataType* __restrict__ output_data,
                                             float* m_scale,
                                             float* n_scale)
{
    // Define epilogue stages for chainer
    using InitEpilogue = CShuffleEpilogueStageBase<Problem>;

    using MainEpilogues = std::conditional_t<UseScale,
                                             ck_tile::tuple<SliceEpilogue<Problem>,
                                                            ScaleEpilogue<Problem>,
                                                            CastLdsEpilogue<Problem>,
                                                            PrepCTensorEpilogue<Problem>,
                                                            ApplyDEpilogue<Problem>,
                                                            StoreToDramEpilogue<Problem>,
                                                            MoveWindowsEpilogue<Problem>>,
                                             ck_tile::tuple<SliceEpilogue<Problem>,
                                                            CastLdsEpilogue<Problem>,
                                                            PrepCTensorEpilogue<Problem>,
                                                            ApplyDEpilogue<Problem>,
                                                            StoreToDramEpilogue<Problem>,
                                                            MoveWindowsEpilogue<Problem>>>;

    using Epilogue = EpilogueChainer<InitEpilogue, MainEpilogues>;

    static_assert(Problem::kMPerBlock <= M && Problem::kNPerBlock <= N,
                  "Block size must fit in tensor dimensions");

    // Allocate shared memory for epilogue
    __shared__ char smem[Epilogue::GetSmemSize()];

    // Create accumulator tile
    constexpr auto lds_distribution_encode =
        make_static_tile_distribution(Epilogue::MakeLdsDistributionEncode());
    auto acc_tile =
        make_static_distributed_tensor<typename Epilogue::AccDataType>(lds_distribution_encode);

    // Fill acc_tile with a simple pattern
    auto& acc_buffer = acc_tile.get_thread_buffer();
    acc_buffer[0]    = 2.0F;

    // Create output tensor view
    auto output_tensor_view =
        make_naive_tensor_view<address_space_enum::global>(output_data,
                                                           make_tuple(M, N),
                                                           make_tuple(N, 1),
                                                           number<Epilogue::GetVectorSizeC()>{},
                                                           number<1>{});

    // Create output tile window
    auto output_tile_window =
        make_tile_window(output_tensor_view,
                         make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}),
                         {0, 0});

    // Create empty D tensors tuple
    auto empty_ds = make_tuple();

    // Call the epilogue chainer
    if constexpr(UseScale)
    {
        const auto m_scale_window = make_tile_window(
            make_naive_tensor_view<address_space_enum::global>(
                m_scale, make_tuple(M, N), make_tuple(1, 0), number<1>{}, number<1>{}),
            make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}),
            {0, 0});
        const auto n_scale_window = make_tile_window(
            make_naive_tensor_view<address_space_enum::global>(
                n_scale, make_tuple(M, N), make_tuple(0, 1), number<1>{}, number<1>{}),
            make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}),
            {0, 0});

        auto init_args = make_tuple();
        auto main_args =
            make_tuple(make_tuple(),                               // SliceEpilogue args
                       make_tuple(m_scale_window, n_scale_window), // ScaleEpilogue args
                       make_tuple(),                               // CastLdsEpilogue args
                       make_tuple(),                               // PrepCTensorEpilogue args
                       make_tuple(),                               // ApplyDEpilogue args
                       make_tuple(),                               // StoreToDramEpilogue args
                       make_tuple()                                // MoveWindowsEpilogue args
            );
        auto final_args = make_tuple();

        Epilogue{}(output_tile_window,
                   acc_tile,
                   empty_ds,
                   smem,
                   init_args,
                   main_args,
                   final_args,
                   std::true_type{});
    }
    else
    {
        Epilogue{}(output_tile_window, acc_tile, empty_ds, smem, std::true_type{});
    }
}

// Test configuration helper - reuse the same problem type
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename ODataType,
          index_t kM,
          index_t kN,
          index_t MWave,
          index_t NWave,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t KPerXdl>
using SimpleEpilogueChainerProblem =
    CShuffleEpilogueStageProblem<ADataType,
                                 BDataType,
                                 ck_tile::tuple<>, // Empty Ds datatype tuple
                                 AccDataType,
                                 ODataType,
                                 ck_tile::tuple<>,                   // Empty Ds layout
                                 tensor_layout::gemm::RowMajor,      // ELayout
                                 ck_tile::element_wise::PassThrough, // CDElementwise
                                 kM,
                                 kN,
                                 MWave,
                                 NWave,
                                 MPerXdl,
                                 NPerXdl,
                                 KPerXdl,
                                 false, // isCTransposed,
                                 memory_operation_enum::set>;

template <typename Problem, index_t M, index_t N>
bool run_epilogue_chainer_test(bool use_scale = false)
{
    using ODataType = typename Problem::ODataType;

    constexpr index_t kMPerBlock = Problem::kMPerBlock;
    constexpr index_t kNPerBlock = Problem::kNPerBlock;
    constexpr index_t kBlockSize = Problem::kBlockSize;

    std::cout << "Running EpilogueChainer test with M=" << M << ", N=" << N
              << ", MPerBlock=" << kMPerBlock << ", NPerBlock=" << kNPerBlock
              << ", BlockSize=" << kBlockSize << std::endl;

    // Allocate host memory
    const size_t output_size = M * N;
    std::vector<ODataType> host_output(output_size, static_cast<ODataType>(0));

    // Allocate device memory
    ODataType* device_output;
    HIP_CHECK_ERROR(hipMalloc(&device_output, output_size * sizeof(ODataType)));
    HIP_CHECK_ERROR(hipMemcpy(
        device_output, host_output.data(), output_size * sizeof(ODataType), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(kBlockSize, 1, 1);

    if(use_scale)
    {
        float* m_scale;
        float* n_scale;
        std::vector<float> h_m_scale(M, 1.0F);
        std::vector<float> h_n_scale(N, 1.0F);
        h_n_scale[1] = 2.0F; // multiply one col only with 2
        HIP_CHECK_ERROR(hipMalloc(&m_scale, M * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&n_scale, N * sizeof(float)));
        HIP_CHECK_ERROR(
            hipMemcpy(m_scale, h_m_scale.data(), M * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK_ERROR(
            hipMemcpy(n_scale, h_n_scale.data(), N * sizeof(float), hipMemcpyHostToDevice));

        test_epilogue_chainer_kernel<Problem, M, N, true>
            <<<gridSize, blockSize>>>(device_output, m_scale, n_scale);

        HIP_CHECK_ERROR(hipFree(m_scale));
        HIP_CHECK_ERROR(hipFree(n_scale));
    }
    else
    {
        test_epilogue_chainer_kernel<Problem, M, N, false>
            <<<gridSize, blockSize>>>(device_output, nullptr, nullptr);
    }

    // Check for kernel launch errors
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Copy results back
    HIP_CHECK_ERROR(hipMemcpy(
        host_output.data(), device_output, output_size * sizeof(ODataType), hipMemcpyDeviceToHost));

    // Basic verification - just check that output has a 2, and 4 if using scaling
    bool has_2 =
        type_convert<float>(host_output[0]) > 1.9F && type_convert<float>(host_output[0]) < 2.1F;
    bool scale_has_4 = true;
    if(use_scale)
    {
        scale_has_4 = type_convert<float>(host_output[1]) > 3.9F &&
                      type_convert<float>(host_output[1]) < 4.1F;
    }

    // Cleanup
    HIP_CHECK_ERROR(hipFree(device_output));

    return has_2 && scale_has_4;
}

} // namespace ck_tile
