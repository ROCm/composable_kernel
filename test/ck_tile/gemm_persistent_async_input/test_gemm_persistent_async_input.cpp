// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/core/utility/persistent_async_input_scheduler.hpp"

#include <chrono>
#include <thread>

using Row       = ck_tile::tensor_layout::gemm::RowMajor;
using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
using F16       = ck_tile::fp16_t;
using F32       = ck_tile::fp32_t;
using Intrawave = ck_tile::integral_constant<ck_tile::GemmPipelineScheduler,
                                             ck_tile::GemmPipelineScheduler::Intrawave>;

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType>
class TestGemmPersistentAsyncInput : public ::testing::Test
{
    protected:
    // Use larger M to ensure tiles_m > tile_idx_pivot, exercising the async scheduler
    static constexpr ck_tile::index_t M = 1536; // 6 tiles with M_Tile=256
    static constexpr ck_tile::index_t N = 1024;
    static constexpr ck_tile::index_t K = 512;

    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    template <bool IsRowMajor>
    static constexpr ck_tile::index_t get_default_stride(ck_tile::index_t row, ck_tile::index_t col)
    {
        if constexpr(IsRowMajor)
            return col;
        else
            return row;
    }

    void Run()
    {
        constexpr bool is_a_row_major = std::is_same_v<ALayout, Row>;
        constexpr bool is_b_row_major = std::is_same_v<BLayout, Row>;
        constexpr bool is_c_row_major = std::is_same_v<CLayout, Row>;

        ck_tile::index_t stride_A = get_default_stride<is_a_row_major>(M, K);
        ck_tile::index_t stride_B = get_default_stride<is_b_row_major>(K, N);
        ck_tile::index_t stride_C = get_default_stride<is_c_row_major>(M, N);

        ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
            M, K, stride_A, ck_tile::bool_constant<is_a_row_major>{}));
        ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
            K, N, stride_B, ck_tile::bool_constant<is_b_row_major>{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(ck_tile::host_tensor_descriptor(
            M, N, stride_C, ck_tile::bool_constant<is_c_row_major>{}));
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
            M, N, stride_C, ck_tile::bool_constant<is_c_row_major>{}));

        // Fill input tensors with random values
        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, 11940}(b_k_n);

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        // Copy input data to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();
        c_m_n_host_ref.SetZero();

        // Compute reference result on host
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);

        // Setup kernel configuration for persistent async input GEMM
        constexpr int kBlockPerCu                          = 1;
        constexpr bool kPadM                               = true;
        constexpr bool kPadN                               = true;
        constexpr bool kPadK                               = true;
        constexpr bool DoubleSmemBuffer                    = true;
        constexpr bool TransposeC                          = false;
        constexpr bool StructuredSparsity                  = false;
        constexpr bool Persistent                          = true;
        constexpr int NumWaveGroup                         = 1;
        constexpr bool Preshuffle                          = false;
        constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
        constexpr ck_tile::index_t TilePartitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                                           TilePartitionerGroupNum,
                                                                           TilePartitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC,
                                                                     StructuredSparsity,
                                                                     Persistent,
                                                                     NumWaveGroup,
                                                                     Preshuffle>;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           Intrawave::value>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompAsync<UniversalGemmProblem>;

        using DsLayout   = ck_tile::tuple<>;
        using DsDataType = ck_tile::tuple<>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             1,     /*kNumWaveGroups_*/
                                             false, /*FixedVectorSize_*/
                                             1,     /*VectorSizeC_*/
                                             1,     /*BlockedXDLN_PerWarp_*/
                                             DoubleSmemBuffer /*DoubleSmemBuffer*/>>;

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        // Calculate tiles and chunks for async scheduler.
        // Uses modulo wraparound like PyTorch - chunk_idx = (iM + pivot) / tiles_per_chunk %
        // num_chunks
        constexpr ck_tile::index_t tiles_per_chunk = 2;
        constexpr ck_tile::index_t tile_idx_pivot  = 2;

        const ck_tile::index_t tiles_m = ck_tile::integer_divide_ceil(M, M_Tile);
        // With add logic, max chunk_idx = (tiles_m - 1 + pivot) / tiles_per_chunk
        // So num_chunks = ceil((tiles_m + pivot) / tiles_per_chunk)
        const ck_tile::index_t num_chunks =
            ck_tile::integer_divide_ceil(tiles_m + tile_idx_pivot, tiles_per_chunk);

        // Validate async scheduler configuration
        // With M=1536, M_Tile=256: tiles_m=6, num_chunks=ceil((6+2)/2)=4
        ASSERT_GT(num_chunks, 0) << "Test requires num_chunks > 0 to exercise async scheduler";
        ASSERT_GT(tiles_per_chunk, 0) << "tiles_per_chunk must be positive";

        // Allocate chunk signals (initialized to zero)
        ck_tile::DeviceMem signal_buf(num_chunks * sizeof(uint32_t));
        signal_buf.SetZero();
        uint32_t* d_chunk_signals = static_cast<uint32_t*>(signal_buf.GetDeviceBuffer());
        ASSERT_NE(d_chunk_signals, nullptr) << "Failed to allocate signal buffer";

        // Setup async input scheduler
        ck_tile::PersistentAsyncInputScheduler async_scheduler;
        async_scheduler.tiles_per_chunk_m = tiles_per_chunk;
        async_scheduler.chunk_signals     = d_chunk_signals;
        async_scheduler.tile_idx_pivot_m  = tile_idx_pivot;
        async_scheduler.num_chunks        = num_chunks;

        // Create UniversalGemmHostArgs with async scheduler
        ck_tile::UniversalGemmHostArgs<1, 1, 0> host_args({a_m_k_dev_buf.GetDeviceBuffer()},
                                                          {b_k_n_dev_buf.GetDeviceBuffer()},
                                                          {},
                                                          c_m_n_dev_buf.GetDeviceBuffer(),
                                                          1, // k_batch
                                                          M,
                                                          N,
                                                          K,
                                                          {stride_A},
                                                          {stride_B},
                                                          {},
                                                          stride_C,
                                                          async_scheduler);

        // Create kernel args using UniversalGemmKernel
        auto kargs = Kernel::UniversalGemmKernel::MakeKernelArgs(host_args);

        // Validate kernel args match host configuration
        ASSERT_EQ(kargs.async_input_scheduler.chunk_signals, d_chunk_signals)
            << "Kernel args chunk_signals doesn't match host configuration";
        ASSERT_EQ(kargs.async_input_scheduler.tiles_per_chunk_m,
                  static_cast<uint32_t>(tiles_per_chunk))
            << "Kernel args tiles_per_chunk_m doesn't match host configuration";
        ASSERT_EQ(kargs.async_input_scheduler.tile_idx_pivot_m,
                  static_cast<int32_t>(tile_idx_pivot))
            << "Kernel args tile_idx_pivot_m doesn't match host configuration";

        // Setup grid and blocks for persistent kernel
        ck_tile::stream_config stream_cfg{nullptr, false};
        const dim3 grids  = Kernel::MaxOccupancyGridSize(stream_cfg);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            GTEST_SKIP() << "Kernel arguments not supported, skipping test";
            return;
        }

        // Create a separate stream for setting signals
        // Using the same stream would deadlock - memcpy waits for kernel, kernel waits for signal
        hipStream_t signal_stream;
        HIP_CHECK_ERROR(hipStreamCreateWithFlags(&signal_stream, hipStreamNonBlocking));

        // Launch kernel
        ck_tile::ignore = ck_tile::launch_kernel(
            stream_cfg, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        // Simulate producer setting chunk signals with interleaved sleep
        // This simulates async input becoming available over time
        const int sleep_us = 100; // microseconds between chunks
        for(ck_tile::index_t i = 0; i < num_chunks; ++i)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            const uint32_t signal_val = 1;
            HIP_CHECK_ERROR(hipMemcpyAsync(d_chunk_signals + i,
                                           &signal_val,
                                           sizeof(uint32_t),
                                           hipMemcpyHostToDevice,
                                           signal_stream));
        }

        // Wait for all signals to be set
        HIP_CHECK_ERROR(hipStreamSynchronize(signal_stream));
        HIP_CHECK_ERROR(hipStreamDestroy(signal_stream));

        // Wait for kernel completion
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        // Copy result back to host
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

        // Validate results
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());

        const auto rtol = ck_tile::get_relative_threshold<ADataType, CDataType, AccDataType>(K);
        const auto atol = ck_tile::get_absolute_threshold<ADataType, CDataType, AccDataType>(
            max_accumulated_value, K);

        bool pass = ck_tile::check_err(
            c_m_n_dev_result, c_m_n_host_ref, "Error: Incorrect results!", rtol, atol);

        EXPECT_TRUE(pass);
    }
};

// Define test types for different layout combinations
using RowRowRow_F16F16F32F16 = TestGemmPersistentAsyncInput<Row, Row, Row, F16, F16, F32, F16>;
using RowColRow_F16F16F32F16 = TestGemmPersistentAsyncInput<Row, Col, Row, F16, F16, F32, F16>;
using ColRowRow_F16F16F32F16 = TestGemmPersistentAsyncInput<Col, Row, Row, F16, F16, F32, F16>;
using ColColRow_F16F16F32F16 = TestGemmPersistentAsyncInput<Col, Col, Row, F16, F16, F32, F16>;

// Test case for Row-Row-Row layout
TEST_F(RowRowRow_F16F16F32F16, BasicTest) { this->Run(); }

// Test case for Row-Col-Row layout
TEST_F(RowColRow_F16F16F32F16, BasicTest) { this->Run(); }

// Test case for Col-Row-Row layout
TEST_F(ColRowRow_F16F16F32F16, BasicTest) { this->Run(); }

// Test case for Col-Col-Row layout
TEST_F(ColColRow_F16F16F32F16, BasicTest) { this->Run(); }
