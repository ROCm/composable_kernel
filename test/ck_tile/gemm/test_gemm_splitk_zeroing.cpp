// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <tuple>
#include <cmath>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "test_gemm_pipeline_util.hpp"

// Copy the exact pipeline configuration from gemm_utils.hpp - only define what we use
#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2

#ifndef CK_TILE_PIPELINE_DEFAULT
#define CK_TILE_PIPELINE_DEFAULT CK_TILE_PIPELINE_COMPUTE_V3
#endif

#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrMem
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Interwave
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV3
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave
#else
#error "unsupported CK_TILE_PIPELINE_DEFAULT value"
#endif

namespace splitk_zeroing_test {
    // Use exact same configuration as GemmConfig from gemm_utils.hpp
    struct TestGemmConfig
    {
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
        // Compute friendly for Intrawave scheduler
        static constexpr ck_tile::index_t M_Tile = 128;
        static constexpr ck_tile::index_t N_Tile = 128;
        static constexpr ck_tile::index_t K_Tile = 128;

        static constexpr ck_tile::index_t M_Warp = 2;
        static constexpr ck_tile::index_t N_Warp = 2;
        static constexpr ck_tile::index_t K_Warp = 1;

        static constexpr ck_tile::index_t M_Warp_Tile = 16;
        static constexpr ck_tile::index_t N_Warp_Tile = 16;
        static constexpr ck_tile::index_t K_Warp_Tile = 32;

        static constexpr bool DoubleSmemBuffer = false;
#endif

        static constexpr bool kPadM = false;
        static constexpr bool kPadN = false;
        static constexpr bool kPadK = false;

        static constexpr bool PermuteA = false;
        static constexpr bool PermuteB = false;

        static constexpr bool TransposeC            = false;
        static constexpr bool UseStructuredSparsity = false;

        static constexpr int kBlockPerCu                         = 1;
        static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        static constexpr ck_tile::index_t TileParitionerM01      = 4;
    };

    // Copy the exact calculate_rtol_atol function from run_gemm_example.inc
    template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
    auto calculate_split_k_rtol_atol(const ck_tile::index_t K,
                                     const ck_tile::index_t kbatch,
                                     const float max_accumulated_value)
    {
        using ComputeType =
            std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
        // Calculate thresholds
        const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
            ck_tile::integer_divide_ceil(K, kbatch));
        const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
            max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
        // Calculate error due to split_k accumulation
        const auto rtol_split_k =
            ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
        const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
            max_accumulated_value, kbatch);
        // Use higher threshold
        return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
    }

    // Copy helper functions exactly from universal_gemm.cpp
    template<typename Layout>
    static constexpr inline auto is_row_major(Layout layout_)
    {
        return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                     ck_tile::tensor_layout::gemm::RowMajor>>{};
    }

    // Calculate number of tiles for barrier allocation - exactly from universal_gemm.cpp
    template<typename TilePartitioner>
    auto CalculateNumTiles(const ck_tile::GemmHostArgs& args)
    {
        const auto M_blocks = (args.M + TilePartitioner::MPerBlock - 1) / TilePartitioner::MPerBlock;
        const auto N_blocks = (args.N + TilePartitioner::NPerBlock - 1) / TilePartitioner::NPerBlock;
        return M_blocks * N_blocks;
    }

    // Allocate barriers for Split-K synchronization - exactly from universal_gemm.cpp
    template<typename TilePartitioner>
    std::pair<uint32_t*, uint32_t*> AllocateSplitKBarriers(const ck_tile::GemmHostArgs& args)
    {
        if(args.k_batch <= 1) {
            return std::make_pair(nullptr, nullptr);
        }
        
        const auto total_tiles = CalculateNumTiles<TilePartitioner>(args);
        const size_t barrier_size = total_tiles * sizeof(uint32_t);
        
        uint32_t* cleared_c_tile_barrier;
        uint32_t* updated_batches_barrier;
        
        hipError_t hip_err = hipMalloc(&cleared_c_tile_barrier, barrier_size);
        if(hip_err != hipSuccess) {
            throw std::runtime_error("Failed to allocate cleared_c_tile_barrier: " + 
                                std::string(hipGetErrorString(hip_err)));
        }
        
        hip_err = hipMalloc(&updated_batches_barrier, barrier_size);
        if(hip_err != hipSuccess) {
            (void)hipFree(cleared_c_tile_barrier);
            throw std::runtime_error("Failed to allocate updated_batches_barrier: " + 
                                std::string(hipGetErrorString(hip_err)));
        }
        
        hip_err = hipMemset(cleared_c_tile_barrier, 0, barrier_size);
        if(hip_err != hipSuccess) {
            (void)hipFree(cleared_c_tile_barrier);
            (void)hipFree(updated_batches_barrier);
            throw std::runtime_error("Failed to initialize cleared_c_tile_barrier");
        }
        
        hip_err = hipMemset(updated_batches_barrier, 0, barrier_size);
        if(hip_err != hipSuccess) {
            (void)hipFree(cleared_c_tile_barrier);
            (void)hipFree(updated_batches_barrier);
            throw std::runtime_error("Failed to initialize updated_batches_barrier");
        }
        
        return std::make_pair(cleared_c_tile_barrier, updated_batches_barrier);
    }

    // Cleanup barriers - exactly from universal_gemm.cpp
    void CleanupSplitKBarriers(uint32_t* cleared_barrier, uint32_t* updated_barrier)
    {
        if(cleared_barrier) {
            (void)hipFree(cleared_barrier);
        }
        if(updated_barrier) {
            (void)hipFree(updated_barrier);
        }
    }
} // end namespace splitk_zeroing_test

template <typename Tuple>
class TestCkTileGemmSplitKZeroing : public ::testing::Test
{
    public:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using CDataType   = std::tuple_element_t<3, Tuple>;
    using ALayout     = std::tuple_element_t<4, Tuple>;
    using BLayout     = std::tuple_element_t<5, Tuple>;
    using CLayout     = std::tuple_element_t<6, Tuple>;

    protected:
    void RunZeroingTest(ck_tile::index_t M, 
                        ck_tile::index_t N, 
                        ck_tile::index_t K, 
                        ck_tile::index_t k_batch,
                        bool initialize_c_nonzero = true)
    {
        // Use exact same stride calculation as run_gemm_example.inc
        ck_tile::index_t stride_A = ck_tile::get_default_stride(M, K, 0, splitk_zeroing_test::is_row_major(ALayout{}));
        ck_tile::index_t stride_B = ck_tile::get_default_stride(K, N, 0, splitk_zeroing_test::is_row_major(BLayout{}));
        ck_tile::index_t stride_C = ck_tile::get_default_stride(M, N, 0, splitk_zeroing_test::is_row_major(CLayout{}));

        // Create host tensors exactly like run_gemm_example.inc
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, splitk_zeroing_test::is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, splitk_zeroing_test::is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> c_m_n_device_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, splitk_zeroing_test::is_row_major(CLayout{})));

        // Initialize input tensors - use same pattern as run_gemm_example.inc
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
        
        if(initialize_c_nonzero) {
            // Initialize C with non-zero values to test zeroing capability
            ck_tile::FillUniformDistribution<CDataType>{-2.f, 2.f}(c_m_n_device_result);
        } else {
            // Initialize C with zeros
            c_m_n_device_result.SetZero();
        }

        // Device memory allocation
        ck_tile::DeviceMem a_device_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_device_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_device_buf(c_m_n_device_result.get_element_space_size_in_bytes());

        a_device_buf.ToDevice(a_m_k.data());
        b_device_buf.ToDevice(b_k_n.data());
        c_device_buf.ToDevice(c_m_n_device_result.data());

        std::cout << "Matrix dimensions: M=" << M << " N=" << N << " K=" << K << " k_batch=" << k_batch << std::endl;
        std::cout << "Strides: A=" << stride_A << " B=" << stride_B << " C=" << stride_C << std::endl;

        // Use EXACT same template structure as universal_gemm.cpp
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<splitk_zeroing_test::TestGemmConfig::M_Tile, splitk_zeroing_test::TestGemmConfig::N_Tile, splitk_zeroing_test::TestGemmConfig::K_Tile>,
            ck_tile::sequence<splitk_zeroing_test::TestGemmConfig::M_Warp, splitk_zeroing_test::TestGemmConfig::N_Warp, splitk_zeroing_test::TestGemmConfig::K_Warp>,
            ck_tile::sequence<splitk_zeroing_test::TestGemmConfig::M_Warp_Tile, splitk_zeroing_test::TestGemmConfig::N_Warp_Tile, splitk_zeroing_test::TestGemmConfig::K_Warp_Tile>,
            splitk_zeroing_test::TestGemmConfig::PermuteA,
            splitk_zeroing_test::TestGemmConfig::PermuteB>;

        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                                           splitk_zeroing_test::TestGemmConfig::TileParitionerGroupNum,
                                                                           splitk_zeroing_test::TestGemmConfig::TileParitionerM01>;

        using Traits = ck_tile::TileGemmTraits<splitk_zeroing_test::TestGemmConfig::kPadM,
                                               splitk_zeroing_test::TestGemmConfig::kPadN,
                                               splitk_zeroing_test::TestGemmConfig::kPadK,
                                               ALayout,
                                               BLayout,
                                               CLayout>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<splitk_zeroing_test::TestGemmConfig::kPadM,
                                                                     splitk_zeroing_test::TestGemmConfig::kPadN,
                                                                     splitk_zeroing_test::TestGemmConfig::kPadK,
                                                                     splitk_zeroing_test::TestGemmConfig::DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     splitk_zeroing_test::TestGemmConfig::TransposeC,
                                                                     splitk_zeroing_test::TestGemmConfig::UseStructuredSparsity,
                                                                     false>; // Not persistent

        using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        // Use simplified pipeline similar to universal_gemm
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           GEMM_PIPELINE_SCHEDULER,
                                                                           true, // has_hot_loop
                                                                           ck_tile::TailNumber::Full>;

        using GemmPipeline = GEMM_PIPELINE<UniversalGemmProblem>;

        // Use CShuffleEpilogue exactly like universal_gemm.cpp
        constexpr bool UseZeroing = true;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             AccDataType,
                                             CDataType,
                                             CLayout,
                                             GemmPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             splitk_zeroing_test::TestGemmConfig::M_Warp,
                                             splitk_zeroing_test::TestGemmConfig::N_Warp,
                                             splitk_zeroing_test::TestGemmConfig::M_Warp_Tile,
                                             splitk_zeroing_test::TestGemmConfig::N_Warp_Tile,
                                             splitk_zeroing_test::TestGemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             ck_tile::memory_operation_enum::atomic_add>>;

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, UseZeroing>;

        // Create host args structure exactly like universal_gemm.cpp
        ck_tile::GemmHostArgs args;
        args.a_ptr    = a_device_buf.GetDeviceBuffer();
        args.b_ptr    = b_device_buf.GetDeviceBuffer();
        args.c_ptr    = c_device_buf.GetDeviceBuffer();
        args.k_batch  = k_batch;
        args.M        = M;
        args.N        = N;
        args.K        = K;
        args.stride_A = stride_A;
        args.stride_B = stride_B;
        args.stride_C = stride_C;

        // Allocate barriers exactly like universal_gemm.cpp
        auto [cleared_barrier, updated_barrier] = splitk_zeroing_test::AllocateSplitKBarriers<TilePartitioner>(args);

        // Create kernel arguments exactly like universal_gemm.cpp
        auto kargs = Kernel::MakeKernelArgs(args);
        kargs.cleared_c_tile_barrier = cleared_barrier;
        kargs.updated_batches_barrier = updated_barrier;

        // Check if kernel supports the arguments BEFORE launching
        if(!Kernel::IsSupportedArgument(kargs)) {
            splitk_zeroing_test::CleanupSplitKBarriers(cleared_barrier, updated_barrier);
            GTEST_SKIP() << "Kernel configuration not supported for M=" << M << " N=" << N << " K=" << K;
        }

        // Launch kernel exactly like universal_gemm.cpp
        const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        std::cout << "Grid: (" << grids.x << "," << grids.y << "," << grids.z << ")" << std::endl;
        std::cout << "Block: (" << blocks.x << "," << blocks.y << "," << blocks.z << ")" << std::endl;

        float kernel_time = ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, true, 0},
            ck_tile::make_kernel<blocks.x, splitk_zeroing_test::TestGemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        // Copy result back
        c_device_buf.FromDevice(c_m_n_device_result.data());

        // Compute reference result exactly like run_gemm_example.inc
        ck_tile::HostTensor<CDataType> c_m_n_reference(
            ck_tile::host_tensor_descriptor(M, N, stride_C, splitk_zeroing_test::is_row_major(CLayout{})));
        c_m_n_reference.SetZero();
        
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_reference);

        // Calculate error tolerances exactly like run_gemm_example.inc
        const float max_accumulated_value =
            *std::max_element(c_m_n_reference.mData.begin(), c_m_n_reference.mData.end());
        const auto rtol_atol = splitk_zeroing_test::calculate_split_k_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, k_batch, max_accumulated_value);

        // Verify results exactly like run_gemm_example.inc
        bool pass = ck_tile::check_err(c_m_n_device_result,
                                      c_m_n_reference,
                                      "Error: Incorrect results!",
                                      rtol_atol.at(ck_tile::number<0>{}),
                                      rtol_atol.at(ck_tile::number<1>{}));

        // Debug output
        std::cout << "Split-K Zeroing Test: M=" << M << ", N=" << N << ", K=" << K 
                  << ", k_batch=" << k_batch << ", C_init=" << (initialize_c_nonzero ? "nonzero" : "zero")
                  << " -> " << (pass ? "PASS" : "FAIL") << " (time: " << kernel_time << "ms)" << std::endl;
        
        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;

        // Cleanup
        splitk_zeroing_test::CleanupSplitKBarriers(cleared_barrier, updated_barrier);

        EXPECT_TRUE(pass) << "Split-K GEMM with zeroing failed for M=" << M 
                          << ", N=" << N << ", K=" << K << ", k_batch=" << k_batch;
    }
};

// Test with same type as universal_gemm supports
using TestTypes = ::testing::Types<
    std::tuple<ck_tile::half_t, ck_tile::half_t, float, ck_tile::half_t,
               ck_tile::tensor_layout::gemm::RowMajor, 
               ck_tile::tensor_layout::gemm::ColumnMajor, 
               ck_tile::tensor_layout::gemm::RowMajor>
>;

TYPED_TEST_SUITE(TestCkTileGemmSplitKZeroing, TestTypes);

TYPED_TEST(TestCkTileGemmSplitKZeroing, SmallMatrixZeroingTest)
{
    // Test small matrices with split-K
    this->RunZeroingTest(256, 256, 512, 2, true);   // C starts non-zero
}

TYPED_TEST(TestCkTileGemmSplitKZeroing, MediumMatrixZeroingTest)
{
    // Test medium matrices
    this->RunZeroingTest(512, 512, 512, 2, true); 
}

TYPED_TEST(TestCkTileGemmSplitKZeroing, PreZeroedComparisonTest)
{
    // Verify pre-zeroed C gives same result as auto-zeroed C
    this->RunZeroingTest(256, 256, 512, 4, false); // C starts zero
}
