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
#include <array>
#include <iostream>
#include <utility>
#include <vector>
#include <hip/hip_runtime.h>

namespace ck_tile {

enum class ScaleType
{
    None,
    RowCol,
    Tensor
};

// Simple test kernel to invoke the CShuffleEpilogue
template <typename Problem, index_t M, index_t N, ScaleType Scale>
__global__ void
test_cshuffle_epilogue_kernel(const typename Problem::AccDataType* __restrict__ input_data,
                              typename Problem::ODataType* __restrict__ output_data,
                              float* m_scale,
                              float* n_scale)
{
    using Epilogue    = CShuffleEpilogue<Problem>;
    using AccDataType = typename Epilogue::AccDataType;

    static_assert(Problem::kMPerBlock <= M && Problem::kNPerBlock <= N,
                  "Block size must fit in tensor dimensions");

    // Allocate shared memory for epilogue
    __shared__ char smem[Epilogue::GetSmemSize()];

    // Create accumulator tile with GEMM accumulator distribution (matches BlockGemm)
    using WG = ck_tile::WarpGemmDispatcher<typename Epilogue::ATypeToUse,
                                           typename Epilogue::BTypeToUse,
                                           typename Problem::AccDataType,
                                           Problem::MPerXdl,
                                           Problem::NPerXdl,
                                           Problem::KPerXdl,
                                           Problem::isCTransposed>;

    constexpr index_t MIterPerWarp = Problem::kMPerBlock / (Problem::MWave * Problem::MPerXdl);
    constexpr index_t NIterPerWarp = Problem::kNPerBlock / (Problem::NWave * Problem::NPerXdl);

    constexpr auto c_block_outer_dstr_encoding = ck_tile::tile_distribution_encoding<
        ck_tile::sequence<>,
        ck_tile::tuple<ck_tile::sequence<MIterPerWarp, Problem::MWave>,
                       ck_tile::sequence<NIterPerWarp, Problem::NWave>>,
        ck_tile::tuple<ck_tile::sequence<1, 2>>,
        ck_tile::tuple<ck_tile::sequence<1, 1>>,
        ck_tile::sequence<1, 2>,
        ck_tile::sequence<0, 0>>{};

    constexpr auto acc_distribution_encode = ck_tile::detail::make_embed_tile_distribution_encoding(
        c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

    constexpr auto acc_distribution = make_static_tile_distribution(acc_distribution_encode);
    auto acc_tile                   = make_static_distributed_tensor<AccDataType>(acc_distribution);

    // Create input tensor view for loading from global memory
    // Note: cast away const since buffer_view infrastructure doesn't support const pointers,
    // but the input_data is only read, never written
    // Use runtime values for dimensions to avoid issues with constant buffer size types
    constexpr index_t kMPerBlock = Problem::kMPerBlock;
    constexpr index_t kNPerBlock = Problem::kNPerBlock;
    auto input_tensor_view       = make_naive_tensor_view<address_space_enum::global>(
        const_cast<AccDataType*>(input_data),
        make_tuple(kMPerBlock, kNPerBlock),
        make_tuple(kNPerBlock, 1), // row-major strides
        number<1>{},
        number<1>{});

    // Create tile window using the correct accumulator distribution
    auto input_tile_window =
        make_tile_window(input_tensor_view,
                         make_tuple(number<Problem::kMPerBlock>{}, number<Problem::kNPerBlock>{}),
                         {0, 0},
                         acc_distribution); // Use GEMM acc distribution, not LDS distribution

    // Load input data from global memory into acc_tile
    load_tile(acc_tile, input_tile_window);

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

// Launch kernel with RowCol scaling
template <typename Problem, index_t M, index_t N>
void launch_kernel_with_rowcol_scale(const typename Problem::AccDataType* device_input,
                                     typename Problem::ODataType* device_output,
                                     dim3 gridSize,
                                     dim3 blockSize)
{
    HostTensor<float> h_m_scale({M});
    HostTensor<float> h_n_scale({N});
    for(index_t i = 0; i < M; ++i)
    {
        h_m_scale.mData[i] = 1.0F;
    }
    for(index_t i = 0; i < N; ++i)
    {
        h_n_scale.mData[i] = 1.0F;
    }
    h_n_scale.mData[1] = 2.0F;

    DeviceMem m_scale_buf(h_m_scale.get_element_space_size_in_bytes());
    DeviceMem n_scale_buf(h_n_scale.get_element_space_size_in_bytes());
    m_scale_buf.ToDevice(h_m_scale.data());
    n_scale_buf.ToDevice(h_n_scale.data());

    test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::RowCol>
        <<<gridSize, blockSize>>>(device_input,
                                  device_output,
                                  static_cast<float*>(m_scale_buf.GetDeviceBuffer()),
                                  static_cast<float*>(n_scale_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}

// Launch kernel with Tensor scaling
template <typename Problem, index_t M, index_t N>
void launch_kernel_with_tensor_scale(const typename Problem::AccDataType* device_input,
                                     typename Problem::ODataType* device_output,
                                     dim3 gridSize,
                                     dim3 blockSize)
{
    HostTensor<float> h_m_scale({1});
    HostTensor<float> h_n_scale({1});
    h_m_scale.mData[0] = 2.0F;
    h_n_scale.mData[0] = 1.0F;

    DeviceMem m_scale_buf(h_m_scale.get_element_space_size_in_bytes());
    DeviceMem n_scale_buf(h_n_scale.get_element_space_size_in_bytes());
    m_scale_buf.ToDevice(h_m_scale.data());
    n_scale_buf.ToDevice(h_n_scale.data());

    test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::Tensor>
        <<<gridSize, blockSize>>>(device_input,
                                  device_output,
                                  static_cast<float*>(m_scale_buf.GetDeviceBuffer()),
                                  static_cast<float*>(n_scale_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}

// Launch kernel without scaling
template <typename Problem, index_t M, index_t N>
void launch_kernel_without_scale(const typename Problem::AccDataType* device_input,
                                 typename Problem::ODataType* device_output,
                                 dim3 gridSize,
                                 dim3 blockSize)
{
    test_cshuffle_epilogue_kernel<Problem, M, N, ScaleType::None>
        <<<gridSize, blockSize>>>(device_input, device_output, nullptr, nullptr);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}

/// Generate N unique fp16 bit patterns from the normal range.
/// Uses positive normals (0x0400-0x7BFF) first, then negative normals (0x8400-0xFBFF).
/// Static asserts if N > 61440 (max unique normal fp16 values).
template <size_t N>
constexpr std::array<uint16_t, N> generate_fp16_bit_patterns()
{
    static_assert(N <= 61440, "N exceeds available unique normal fp16 values");

    std::array<uint16_t, N> result{};
    constexpr uint16_t kPosStart         = 0x0400;
    constexpr uint16_t kNegStart         = 0x8400;
    constexpr size_t kMaxPositiveNormals = 30720;

    for(size_t i = 0; i < N; ++i)
    {
        result[i] = (i < kMaxPositiveNormals)
                        ? static_cast<uint16_t>(kPosStart + i)
                        : static_cast<uint16_t>(kNegStart + (i - kMaxPositiveNormals));
    }
    return result;
}

/// Convert fp16 bit patterns to float values.
/// Performs: uint16_t -> half_t (bit_cast) -> float
template <size_t N>
std::array<float, N> convert_fp16_bits(const std::array<uint16_t, N>& bits)
{
    std::array<float, N> result;
    for(size_t i = 0; i < N; ++i)
    {
        half_t h  = bit_cast<half_t>(bits[i]);
        result[i] = type_convert<float>(h);
    }
    return result;
}

/// Generate unique fp16 values as a HostTensor for permutation testing.
/// Uses layered architecture: bit patterns -> type conversion -> HostTensor.
template <typename AccDataType, index_t Rows, index_t Cols>
HostTensor<AccDataType> generate_unique_fp16_input()
{
    constexpr size_t N = static_cast<size_t>(Rows * Cols);

    constexpr auto bits = generate_fp16_bit_patterns<N>();
    auto values         = convert_fp16_bits(bits);

    HostTensor<AccDataType> host_input({Rows, Cols});
    for(index_t m = 0; m < Rows; ++m)
    {
        for(index_t n = 0; n < Cols; ++n)
        {
            host_input(m, n) = static_cast<AccDataType>(values[static_cast<size_t>(m * Cols + n)]);
        }
    }
    return host_input;
}

template <typename Problem, index_t M, index_t N>
auto run_cshuffle_epilogue_test(ScaleType scale = ScaleType::None)
{
    using AccDataType = typename Problem::AccDataType;
    using ODataType   = typename Problem::ODataType;

    constexpr index_t kMPerBlock = Problem::kMPerBlock;
    constexpr index_t kNPerBlock = Problem::kNPerBlock;
    const index_t kBlockSize = ck_tile::is_wave32() ? Problem::kBlockSize / 2 : Problem::kBlockSize;

    std::cout << "Running CShuffleEpilogue test with M=" << M << ", N=" << N
              << ", MPerBlock=" << kMPerBlock << ", NPerBlock=" << kNPerBlock
              << ", BlockSize=" << kBlockSize << std::endl;

    HostTensor<AccDataType> host_input =
        generate_unique_fp16_input<AccDataType, kMPerBlock, kNPerBlock>();

    // Allocate device input and copy from host
    DeviceMem device_input_buf(host_input.get_element_space_size_in_bytes());
    device_input_buf.ToDevice(host_input.data());
    auto* device_input = static_cast<const AccDataType*>(device_input_buf.GetDeviceBuffer());

    // Allocate host output memory
    HostTensor<ODataType> host_output({M, N});
    host_output.SetZero();

    // Allocate device output memory
    DeviceMem device_output_buf(host_output.get_element_space_size_in_bytes());
    device_output_buf.ToDevice(host_output.data());
    ODataType* device_output = static_cast<ODataType*>(device_output_buf.GetDeviceBuffer());

    // Launch kernel with appropriate scale configuration
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(kBlockSize, 1, 1);

    switch(scale)
    {
    case ScaleType::RowCol:
        launch_kernel_with_rowcol_scale<Problem, M, N>(
            device_input, device_output, gridSize, blockSize);
        break;
    case ScaleType::Tensor:
        launch_kernel_with_tensor_scale<Problem, M, N>(
            device_input, device_output, gridSize, blockSize);
        break;
    case ScaleType::None:
        launch_kernel_without_scale<Problem, M, N>(
            device_input, device_output, gridSize, blockSize);
        break;
    }

    // Copy results back
    device_output_buf.FromDevice(host_output.data());

    return host_output;
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

// Run both unscaled and scaled tests for comparison
// Returns pair of (unscaled_output, scaled_output) host tensors
template <typename Problem, index_t M, index_t N, ScaleType ScaleMode>
auto run_scale_comparison_test()
{
    auto unscaled_output = run_cshuffle_epilogue_test<Problem, M, N>(ScaleType::None);
    auto scaled_output   = run_cshuffle_epilogue_test<Problem, M, N>(ScaleMode);

    return std::make_pair(std::move(unscaled_output), std::move(scaled_output));
}

} // namespace ck_tile
