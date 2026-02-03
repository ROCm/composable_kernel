// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <hip/hip_runtime.h>

namespace ck_tile {

// Verification and test constants
namespace verification {
constexpr float kScaleEpsilon = 0.001F;

// Scale factors used in tests - these must match between allocation and verification
constexpr float kTestScaleFactor =
    2.0F; // Scale factor applied in RowCol (to column 1) and Tensor tests
constexpr float kIdentityScale    = 1.0F; // Identity scale (no change)
constexpr index_t kScaledColIndex = 1;    // Column index that gets scaled in RowCol tests
} // namespace verification

enum class ScaleType
{
    None,
    RowCol,
    Tensor
};

// Simple test kernel to invoke the CShuffleEpilogue
template <typename Problem, index_t M, index_t N, ScaleType Scale>
__global__ void test_cshuffle_epilogue_kernel(typename Problem::ODataType* __restrict__ output_data,
                                              float* m_scale,
                                              float* n_scale)
{
    using Epilogue = CShuffleEpilogue<Problem>;

    static_assert(Problem::kMPerBlock <= M && Problem::kNPerBlock <= N,
                  "Block size must fit in tensor dimensions");

    // Allocate shared memory for epilogue
    __shared__ char smem[Epilogue::GetSmemSize()];

    // Create accumulator tile
    constexpr auto lds_distribution_encode =
        make_static_tile_distribution(Epilogue::MakeLdsDistributionEncode());
    auto acc_tile =
        make_static_distributed_tensor<typename Epilogue::AccDataType>(lds_distribution_encode);

    // Fill acc_tile with unique integer values per thread and buffer index
    // Values are exactly representable in the 32-bit float accumulator
    auto& acc_buffer          = acc_tile.get_thread_buffer();
    const index_t thread_id   = threadIdx.x;
    const index_t buffer_size = acc_buffer.size();
    for(index_t i = 0; i < buffer_size; i++)
    {
        // Generate unique value: 1-based to avoid zero
        const index_t unique_val = thread_id * buffer_size + i + 1;
        acc_buffer[i]            = static_cast<typename Epilogue::AccDataType>(unique_val);
    }

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

    // Create empty D tensors tuple (we're ignoring ds_dram_windows for this test)
    auto empty_ds = make_tuple();

    // Call the epilogue
    if constexpr(Scale == ScaleType::RowCol)
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
        Epilogue{}(output_tile_window, acc_tile, empty_ds, smem, m_scale_window, n_scale_window);
    }
    else if constexpr(Scale == ScaleType::Tensor)
    {
        Epilogue{}(output_tile_window, acc_tile, empty_ds, smem, *m_scale, *n_scale);
    }
    else
    {
        Epilogue{}(output_tile_window, acc_tile, empty_ds, smem);
    }
}

// Test configuration helper
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
using SimpleCShuffleEpilogueProblem =
    CShuffleEpilogueProblem<ADataType,
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
                            false // isCTransposed
                            >;

// Result struct containing output for verification
template <typename ODataType>
struct CShuffleEpilogueTestResult
{
    HostTensor<ODataType> output;
};

// Launch kernel with RowCol scaling
template <typename Problem, index_t M, index_t N>
void launch_kernel_with_rowcol_scale(typename Problem::ODataType* device_output,
                                     dim3 gridSize,
                                     dim3 blockSize)
{
    HostTensor<float> h_m_scale({M});
    HostTensor<float> h_n_scale({N});
    for(index_t i = 0; i < M; ++i)
    {
        h_m_scale.mData[i] = verification::kIdentityScale;
    }
    for(index_t i = 0; i < N; ++i)
    {
        h_n_scale.mData[i] = verification::kIdentityScale;
    }
    h_n_scale.mData[verification::kScaledColIndex] = verification::kTestScaleFactor;

    DeviceMem m_scale_buf(h_m_scale.get_element_space_size_in_bytes());
    DeviceMem n_scale_buf(h_n_scale.get_element_space_size_in_bytes());
    m_scale_buf.ToDevice(h_m_scale.data());
    n_scale_buf.ToDevice(h_n_scale.data());

    test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::RowCol>
        <<<gridSize, blockSize>>>(device_output,
                                  static_cast<float*>(m_scale_buf.GetDeviceBuffer()),
                                  static_cast<float*>(n_scale_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}

// Launch kernel with Tensor scaling
template <typename Problem, index_t M, index_t N>
void launch_kernel_with_tensor_scale(typename Problem::ODataType* device_output,
                                     dim3 gridSize,
                                     dim3 blockSize)
{
    HostTensor<float> h_m_scale({1});
    HostTensor<float> h_n_scale({1});
    h_m_scale.mData[0] = verification::kTestScaleFactor;
    h_n_scale.mData[0] = verification::kIdentityScale;

    DeviceMem m_scale_buf(h_m_scale.get_element_space_size_in_bytes());
    DeviceMem n_scale_buf(h_n_scale.get_element_space_size_in_bytes());
    m_scale_buf.ToDevice(h_m_scale.data());
    n_scale_buf.ToDevice(h_n_scale.data());

    test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::Tensor>
        <<<gridSize, blockSize>>>(device_output,
                                  static_cast<float*>(m_scale_buf.GetDeviceBuffer()),
                                  static_cast<float*>(n_scale_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}

// Launch kernel without scaling
template <typename Problem, index_t M, index_t N>
void launch_kernel_without_scale(typename Problem::ODataType* device_output,
                                 dim3 gridSize,
                                 dim3 blockSize)
{
    test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::None>
        <<<gridSize, blockSize>>>(device_output, nullptr, nullptr);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}

template <typename Problem, index_t M, index_t N>
auto run_cshuffle_epilogue_test(ScaleType scale = ScaleType::None)
{
    using ODataType = typename Problem::ODataType;

    constexpr index_t kMPerBlock = Problem::kMPerBlock;
    constexpr index_t kNPerBlock = Problem::kNPerBlock;
    const index_t kBlockSize = ck_tile::is_wave32() ? Problem::kBlockSize / 2 : Problem::kBlockSize;

    std::cout << "Running CShuffleEpilogue test with M=" << M << ", N=" << N
              << ", MPerBlock=" << kMPerBlock << ", NPerBlock=" << kNPerBlock
              << ", BlockSize=" << kBlockSize << std::endl;

    // Allocate host memory
    HostTensor<ODataType> host_output({M, N});
    host_output.SetZero();

    // Allocate device memory
    DeviceMem device_output_buf(host_output.get_element_space_size_in_bytes());
    device_output_buf.ToDevice(host_output.data());
    ODataType* device_output = static_cast<ODataType*>(device_output_buf.GetDeviceBuffer());

    // Launch kernel with appropriate scale configuration
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(kBlockSize, 1, 1);

    switch(scale)
    {
    case ScaleType::RowCol:
        launch_kernel_with_rowcol_scale<Problem, M, N>(device_output, gridSize, blockSize);
        break;
    case ScaleType::Tensor:
        launch_kernel_with_tensor_scale<Problem, M, N>(device_output, gridSize, blockSize);
        break;
    case ScaleType::None:
        launch_kernel_without_scale<Problem, M, N>(device_output, gridSize, blockSize);
        break;
    }

    // Copy results back
    device_output_buf.FromDevice(host_output.data());

    return CShuffleEpilogueTestResult<ODataType>{std::move(host_output)};
}

// Convert output values to sorted float vector for verification
// Uses float as intermediate to preserve precision for floating-point comparison
template <typename ODataType>
std::vector<float> convert_and_sort_output(const HostTensor<ODataType>& output)
{
    std::vector<float> result;
    result.reserve(output.get_element_size());
    for(size_t i = 0; i < output.get_element_size(); ++i)
    {
        result.push_back(type_convert<float>(output.mData[i]));
    }
    std::sort(result.begin(), result.end());
    return result;
}

// Result pair for scale comparison tests
template <typename ODataType>
struct ScaleComparisonResult
{
    CShuffleEpilogueTestResult<ODataType> unscaled;
    CShuffleEpilogueTestResult<ODataType> scaled;
};

// Run both unscaled and scaled tests for comparison
template <typename Problem, index_t M, index_t N, ScaleType ScaleMode>
auto run_scale_comparison_test()
{
    using ODataType = typename Problem::ODataType;

    auto unscaled = run_cshuffle_epilogue_test<Problem, M, N>(ScaleType::None);
    auto scaled   = run_cshuffle_epilogue_test<Problem, M, N>(ScaleMode);

    return ScaleComparisonResult<ODataType>{std::move(unscaled), std::move(scaled)};
}

} // namespace ck_tile
