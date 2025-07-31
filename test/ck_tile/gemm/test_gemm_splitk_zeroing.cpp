// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <gtest/gtest-typed-test.h>
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

    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool Preshuffle                = false;
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
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// Calculate number of tiles for barrier allocation
template <typename TilePartitioner>
auto CalculateNumTiles(const ck_tile::GemmHostArgs& args)
{
    const auto M_blocks = (args.M + TilePartitioner::MPerBlock - 1) / TilePartitioner::MPerBlock;
    const auto N_blocks = (args.N + TilePartitioner::NPerBlock - 1) / TilePartitioner::NPerBlock;
    printf("MPerBlock=%d, NPerBlock=%d, M_blocks=%d, N_blocks=%d, \n",
           TilePartitioner::MPerBlock,
           TilePartitioner::NPerBlock,
           M_blocks,
           N_blocks);
    return M_blocks * N_blocks;
}

// Calculate workspace size needed for barriers only - same as universal_gemm_zeroing.cpp
template <typename TilePartitioner>
size_t GetWorkspaceSize(const ck_tile::GemmHostArgs& args)
{
    if(args.k_batch <= 1)
    {
        return 0; // No barriers needed
    }

    const auto total_tiles    = CalculateNumTiles<TilePartitioner>(args);
    const size_t barrier_size = 2 * total_tiles * sizeof(uint32_t); // Two barriers

    return barrier_size;
}

// Setup workspace with barriers - same as universal_gemm_zeroing.cpp
template <typename TilePartitioner>
uint32_t* SetupWorkspace(const ck_tile::GemmHostArgs& args, ck_tile::DeviceMem& workspace)
{
    if(args.k_batch <= 1)
    {
        return nullptr;
    }

    // Calculate workspace size for barriers only
    const size_t workspace_size = GetWorkspaceSize<TilePartitioner>(args);

    // Allocate workspace using DeviceMem (handles cleanup automatically)
    workspace.Realloc(workspace_size);
    void* workspace_ptr = workspace.GetDeviceBuffer();

    const auto total_tiles    = CalculateNumTiles<TilePartitioner>(args);
    const size_t barrier_size = total_tiles * sizeof(uint32_t);

    uint32_t* workspace_barriers = static_cast<uint32_t*>(workspace_ptr);

    // Initialize barriers to zero
    hipError_t hip_err = hipMemset(workspace_barriers, 0, 2 * barrier_size);
    if(hip_err != hipSuccess)
    {
        throw std::runtime_error("Failed to initialize workspace_barriers");
    }

    return workspace_barriers;
}

// Common kernel types used by both test classes
template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout>
struct KernelTypes
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TestGemmConfig::M_Tile, TestGemmConfig::N_Tile, TestGemmConfig::K_Tile>,
        ck_tile::sequence<TestGemmConfig::M_Warp, TestGemmConfig::N_Warp, TestGemmConfig::K_Warp>,
        ck_tile::sequence<TestGemmConfig::M_Warp_Tile,
                          TestGemmConfig::N_Warp_Tile,
                          TestGemmConfig::K_Warp_Tile>,
        TestGemmConfig::PermuteA,
        TestGemmConfig::PermuteB>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   TestGemmConfig::TileParitionerGroupNum,
                                                   TestGemmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<TestGemmConfig::kPadM,
                                           TestGemmConfig::kPadN,
                                           TestGemmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           CLayout,
                                           TestGemmConfig::NumWaveGroups>;

    using GemmUniversalTraits =
        ck_tile::TileGemmUniversalTraits<TestGemmConfig::kPadM,
                                         TestGemmConfig::kPadN,
                                         TestGemmConfig::kPadK,
                                         TestGemmConfig::DoubleSmemBuffer,
                                         ALayout,
                                         BLayout,
                                         CLayout,
                                         TestGemmConfig::TransposeC,
                                         TestGemmConfig::UseStructuredSparsity,
                                         false,
                                         TestGemmConfig::NumWaveGroups,
                                         TestGemmConfig::Preshuffle>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                       BDataType,
                                                                       AccDataType,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       GEMM_PIPELINE_SCHEDULER,
                                                                       true,
                                                                       ck_tile::TailNumber::Full>;

    using GemmPipeline = GEMM_PIPELINE<UniversalGemmProblem>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccDataType,
                                         CDataType,
                                         DsLayout,
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         UniversalGemmProblem::kBlockSize,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         TestGemmConfig::M_Warp,
                                         TestGemmConfig::N_Warp,
                                         TestGemmConfig::M_Warp_Tile,
                                         TestGemmConfig::N_Warp_Tile,
                                         TestGemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC,
                                         ck_tile::memory_operation_enum::atomic_add,
                                         TestGemmConfig::NumWaveGroups>>;
    using GemmEpilogue_Zeroing = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccDataType,
                                         CDataType,
                                         DsLayout,
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         UniversalGemmProblem::kBlockSize,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         TestGemmConfig::M_Warp,
                                         TestGemmConfig::N_Warp,
                                         TestGemmConfig::M_Warp_Tile,
                                         TestGemmConfig::N_Warp_Tile,
                                         TestGemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC,
                                         ck_tile::memory_operation_enum::atomic_add,
                                         TestGemmConfig::NumWaveGroups,
                                         false,
                                         1,
                                         true>>;

    using ZeroingKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue_Zeroing>;
    using NonZeroingKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
};
} // end namespace splitk_zeroing_test

// =============================================================================
// MAIN SPLIT-K ZEROING TEST CLASS
// =============================================================================

template <typename Tuple>
class TestCkTileGemmSplitKZeroing : public ::testing::Test
{

    public:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = std::tuple_element_t<2, Tuple>;
    using ADataType   = std::tuple_element_t<3, Tuple>;
    using BDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = std::tuple_element_t<5, Tuple>;
    using CDataType   = std::tuple_element_t<6, Tuple>;

    using DsLayout   = ck_tile::tuple<>;
    using DsDataType = ck_tile::tuple<>;

    using KTypes = splitk_zeroing_test::KernelTypes<ADataType,
                                                    BDataType,
                                                    DsDataType,
                                                    AccDataType,
                                                    CDataType,
                                                    ALayout,
                                                    BLayout,
                                                    DsLayout,
                                                    CLayout>;

    protected:
    void RunZeroingTest(ck_tile::index_t M,
                        ck_tile::index_t N,
                        ck_tile::index_t K,
                        ck_tile::index_t k_batch,
                        bool test_kernel_zeroing = true)
    {
        ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, splitk_zeroing_test::is_row_major(ALayout{}));
        ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, splitk_zeroing_test::is_row_major(BLayout{}));
        ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, splitk_zeroing_test::is_row_major(CLayout{}));

        // Create host tensors exactly like run_gemm_example.inc
        ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
            M, K, stride_A, splitk_zeroing_test::is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
            K, N, stride_B, splitk_zeroing_test::is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> c_m_n_device_result(ck_tile::host_tensor_descriptor(
            M, N, stride_C, splitk_zeroing_test::is_row_major(CLayout{})));

        // Initialize input tensors - use same pattern as run_gemm_example.inc
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);

        // Device memory allocation
        ck_tile::DeviceMem a_device_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_device_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_device_buf(c_m_n_device_result.get_element_space_size_in_bytes());

        a_device_buf.ToDevice(a_m_k.data());
        b_device_buf.ToDevice(b_k_n.data());

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

        if(test_kernel_zeroing)
        {
            // Test 1: Kernel should zero C before computing (ZeroingKernel + non-zero C)
            ck_tile::FillUniformDistribution<CDataType>{-2.f, 2.f}(c_m_n_device_result);
            c_device_buf.ToDevice(c_m_n_device_result.data());

            using TestKernel = typename KTypes::ZeroingKernel;
            RunKernelTest<TestKernel>(
                args, "ZeroingKernel with non-zero C", false, a_m_k, b_k_n); // Pass host tensors
        }
        else
        {
            // Test 2: Normal operation with pre-zeroed C (NonZeroingKernel + zero C)

            using TestKernel       = typename KTypes::NonZeroingKernel;
            bool use_preprocessing = false; //(k_batch > 1); // Clear C for split-K
            RunKernelTest<TestKernel>(args,
                                      "NonZeroingKernel with non_zero C",
                                      use_preprocessing,
                                      a_m_k,
                                      b_k_n,
                                      false); // Pass host tensors
        }
    }

    private:
    template <typename Kernel>
    void RunKernelTest(const ck_tile::GemmHostArgs& args,
                       const std::string& test_name,
                       bool use_preprocessing,
                       const ck_tile::HostTensor<ADataType>& a_host,
                       const ck_tile::HostTensor<BDataType>& b_host,
                       bool expect_pass = true)
    {
        // Use DeviceMem for automatic memory management - same as universal_gemm_zeroing.cpp
        ck_tile::DeviceMem workspace;
        uint32_t* workspace_barriers =
            splitk_zeroing_test::SetupWorkspace<typename KTypes::TilePartitioner>(args, workspace);

        auto kargs = Kernel::MakeKernelArgs(args, workspace_barriers);

        // Check if kernel supports the arguments BEFORE launching
        if(!Kernel::IsSupportedArgument(kargs))
        {
            GTEST_SKIP() << "Kernel configuration not supported for M=" << args.M << " N=" << args.N
                         << " K=" << args.K;
        }

        // Launch kernel with preprocessing
        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        // Create proper stream config
        ck_tile::stream_config stream_cfg{nullptr, true, 0};

        float kernel_time;

        if(use_preprocessing)
        {
            // Need preprocessing to clear C buffer
            auto run_preprocess = [&]() {
                // Clear C memory for split-K operations
                hipError_t hip_err = hipMemsetAsync(
                    args.c_ptr, 0, args.M * args.N * sizeof(CDataType), stream_cfg.stream_id_);
                if(hip_err != hipSuccess)
                {
                    throw std::runtime_error("Failed to clear C buffer: " +
                                             std::string(hipGetErrorString(hip_err)));
                }
            };

            kernel_time = ck_tile::launch_kernel_time_mask(
                stream_cfg,
                run_preprocess,
                ck_tile::make_kernel<blocks.x, splitk_zeroing_test::TestGemmConfig::kBlockPerCu>(
                    Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            // Normal launch without preprocessing

            kernel_time = ck_tile::launch_kernel(
                stream_cfg,
                ck_tile::make_kernel<blocks.x, splitk_zeroing_test::TestGemmConfig::kBlockPerCu>(
                    Kernel{}, grids, blocks, 0, kargs));
        }

        // Create host tensor for device result and copy data back
        ck_tile::HostTensor<CDataType> c_m_n_device_result(ck_tile::host_tensor_descriptor(
            args.M, args.N, args.stride_C, splitk_zeroing_test::is_row_major(CLayout{})));

        // Copy result back from device
        hipError_t hip_err = hipMemcpy(c_m_n_device_result.data(),
                                       args.c_ptr,
                                       args.M * args.N * sizeof(CDataType),
                                       hipMemcpyDeviceToHost);
        if(hip_err != hipSuccess)
        {
            throw std::runtime_error("Failed to copy result from device: " +
                                     std::string(hipGetErrorString(hip_err)));
        }

        // Compute reference result exactly like run_gemm_example.inc
        ck_tile::HostTensor<CDataType> c_m_n_reference(ck_tile::host_tensor_descriptor(
            args.M, args.N, args.stride_C, splitk_zeroing_test::is_row_major(CLayout{})));
        c_m_n_reference.SetZero();

        // Use the passed host tensors for reference computation
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_host, b_host, c_m_n_reference);

        // Calculate error tolerances exactly like run_gemm_example.inc
        const float max_accumulated_value =
            *std::max_element(c_m_n_reference.mData.begin(), c_m_n_reference.mData.end());
        const auto rtol_atol = splitk_zeroing_test::
            calculate_split_k_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                args.K, args.k_batch, max_accumulated_value);

        // Verify results
        bool pass = ck_tile::check_err(c_m_n_device_result,
                                       c_m_n_reference,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        if(expect_pass)
        {
            EXPECT_TRUE(pass) << "Test failed: " << test_name;
        }
        else
        {
            EXPECT_FALSE(pass) << "Test was expected to fail but passed: " << test_name;
        }
    }
};

// TEST TYPE DEFINITIONS
using TestTypes = ::testing::Types<std::tuple<ck_tile::tensor_layout::gemm::RowMajor,    // ALayout
                                              ck_tile::tensor_layout::gemm::ColumnMajor, // BLayout
                                              ck_tile::tensor_layout::gemm::RowMajor,    // CLayout
                                              ck_tile::half_t, // ADataType
                                              ck_tile::half_t, // BDataType
                                              float,           // AccDataType
                                              ck_tile::half_t> // CDataType
                                   >;

// =============================================================================
// SPLIT-K ZEROING TESTS
// =============================================================================

TYPED_TEST_SUITE(TestCkTileGemmSplitKZeroing, TestTypes);

TYPED_TEST(TestCkTileGemmSplitKZeroing, KernelZeroingCapabilityTest1)
{
    // Test that ZeroingKernel properly zeros non-zero C before computing
    this->RunZeroingTest(1024, 512, 2048, 2, true); // test_kernel_zeroing = true
}

TYPED_TEST(TestCkTileGemmSplitKZeroing, NormalOperationTest)
{
    // Test that NonZeroingKernel works correctly with non pre-zeroed C
    this->RunZeroingTest(1024, 512, 2048, 2, false); // test_kernel_zeroing = false
}
