// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Regression test for the multi-tile-per-workgroup LDS race in MXFlatmmKernel's
// persistent path.
//
// The kernel allocates its shared smem once per workgroup, then loops over
// tiles. The epilogue exits with `ds_read`s still in flight while the next
// iteration's pipeline issues `async_load_tile_` writes into the same LDS. On
// gfx1250 these use separate counters (`asynccnt` vs `dscnt`), so without a
// barrier between iterations the async write clobbers bytes a lagging wave's
// `ds_read` is still targeting.
//
// To trigger the bug the kernel must be persistent and total_tiles must exceed
// the persistent grid size so a workgroup processes > 1 tile.

#include "ck_tile/host.hpp"
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"

#include "mx_flatmm_arch_traits.hpp"

namespace {

template <ck_tile::index_t NLane, typename dtype>
auto preShuffleWeight(ck_tile::HostTensor<dtype>& src)
{
    auto src_lengths          = src.get_lengths();
    const int K               = src_lengths[0];
    const int N               = src_lengths[1];
    constexpr int packed_size = ck_tile::numeric_traits<dtype>::PackedSize;

    // fp4/fp6:32 or fp8:16
    int KPack = std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;

    int KLane = ck_tile::get_warp_size() / NLane;
    int K0    = K / (KLane * KPack);

    ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({N * K}, {1}));

    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; k += packed_size)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0    = k / (KLane * KPack);
            int tempk = k % (KLane * KPack);
            int k1    = tempk / KPack;
            int k2    = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            shuffled(outputIndex) = src(k, n);
        }
    }
    return shuffled;
}

using ADataType = ck_tile::fp8_t;
using BDataType = ck_tile::fp8_t;
using CDataType = ck_tile::half_t;
using MXFlatmmArchTraits =
    std::conditional_t<GetCurrentTargetId() == ck_tile::core::arch::TargetId::GFX1250,
                       MXFlatmm_GFX1250_FP8FP8_Traits,
                       MXFlatmm_GFX950_FP8FP8_Traits>;

using FlatmmConfig = typename MXFlatmmArchTraits::Config;
using AccDataType  = float;
using ScaleType    = ck_tile::e8m0_t;

using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

using DsLayout   = ck_tile::tuple<>;
using DsDataType = ck_tile::tuple<>;

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 1;
constexpr int ScaleGranularityK = 32;

using ScaleA = ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>;
using ScaleB = ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>;

void run_persistent_test(ck_tile::index_t M,
                         ck_tile::index_t N,
                         ck_tile::index_t K,
                         int init_method,
                         bool expect_multi_tile = false)
{
    constexpr bool a_row_major = true;
    constexpr bool b_row_major = false; // BLayout is ColumnMajor
    constexpr bool c_row_major = true;

    constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;
    ASSERT_EQ(K % ScaleGranularityK, 0) << "K must be multiple of ScaleGranularityK=32";
    ASSERT_EQ(K % APackedSize, 0) << "K must be multiple of A PackedSize";
    ASSERT_EQ(K % BPackedSize, 0) << "K must be multiple of B PackedSize";

    const ck_tile::index_t stride_A =
        ck_tile::get_default_stride(M, K, 0, ck_tile::bool_constant<a_row_major>{});
    const ck_tile::index_t stride_B =
        ck_tile::get_default_stride(K, N, 0, ck_tile::bool_constant<b_row_major>{});
    const ck_tile::index_t stride_C =
        ck_tile::get_default_stride(M, N, 0, ck_tile::bool_constant<c_row_major>{});

    const auto scale_stride_A = ck_tile::get_default_stride(
        M / ScaleGranularityM, K / ScaleGranularityK, 0, ck_tile::bool_constant<a_row_major>{});
    const auto scale_stride_B = ck_tile::get_default_stride(
        K / ScaleGranularityK, N / ScaleGranularityN, 0, ck_tile::bool_constant<b_row_major>{});

    // --- Host tensors ---
    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, ck_tile::bool_constant<a_row_major>{}));
    ck_tile::HostTensor<BDataType> b_origin_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, ck_tile::bool_constant<b_row_major>{}));
    ck_tile::HostTensor<CDataType> c_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<c_row_major>{}));

    ck_tile::HostTensor<ScaleType> scale_a(
        ck_tile::host_tensor_descriptor(M / ScaleGranularityM,
                                        K / ScaleGranularityK,
                                        scale_stride_A,
                                        ck_tile::bool_constant<a_row_major>{}));
    ck_tile::HostTensor<ScaleType> scale_b(
        ck_tile::host_tensor_descriptor(K / ScaleGranularityK,
                                        N / ScaleGranularityN,
                                        scale_stride_B,
                                        ck_tile::bool_constant<b_row_major>{}));

    if(init_method == 0)
    {
        // Random tensor and scale values
        ck_tile::FillUniformDistribution<>{0.0f, 1.0f}(a_host);
        ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_a);
        ck_tile::FillUniformDistribution<>{-.5f, .5f}(b_origin_host);
        ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_b);
    }
    else if(init_method == 1)
    {
        // Constant tensor and scale values
        ck_tile::FillUniformDistribution<>{2.f, 2.f}(a_host);
        ck_tile::FillUniformDistribution<>{0.5f, 0.5f}(scale_a);
        ck_tile::FillUniformDistribution<>{0.5f, 0.5f}(b_origin_host);
        ck_tile::FillUniformDistribution<>{2.f, 2.f}(scale_b);
    }
    else
    {
        FAIL() << "Unexpected init_method: " << init_method;
    }

    // --- Pre-shuffle B and scales ---
    auto b_shuffled       = preShuffleWeight<MXFlatmmArchTraits::GetNLane()>(b_origin_host);
    auto scale_a_shuffled = MXFlatmmArchTraits::template preShuffleScale<true>(scale_a);
    auto scale_b_shuffled = MXFlatmmArchTraits::template preShuffleScale<false>(scale_b);

    // --- Device buffers ---
    ck_tile::DeviceMem a_dev(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev(b_shuffled.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev(c_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sa_dev(scale_a_shuffled.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sb_dev(scale_b_shuffled.get_element_space_size_in_bytes());

    a_dev.ToDevice(a_host.data());
    b_dev.ToDevice(b_shuffled.data());
    c_host.SetZero();
    c_dev.ToDevice(c_host.data());
    sa_dev.ToDevice(scale_a_shuffled.data());
    sb_dev.ToDevice(scale_b_shuffled.data());

    const auto scale_a_dev_ptr =
        ScaleA{static_cast<ScaleType*>(sa_dev.GetDeviceBuffer()), M / ScaleGranularityM};
    const auto scale_b_dev_ptr =
        ScaleB{static_cast<ScaleType*>(sb_dev.GetDeviceBuffer()), N / ScaleGranularityN};

    // --- ScaleFlatmmHostArgs ---
    ck_tile::ScaleFlatmmHostArgs<ScaleA, ScaleB> args{a_dev.GetDeviceBuffer(),
                                                      b_dev.GetDeviceBuffer(),
                                                      {},
                                                      c_dev.GetDeviceBuffer(),
                                                      /*k_batch=*/1,
                                                      M,
                                                      N,
                                                      K,
                                                      stride_A,
                                                      stride_B,
                                                      {},
                                                      stride_C,
                                                      scale_a_dev_ptr,
                                                      scale_b_dev_ptr};

    // --- Kernel type tower (Persistent=true enables the persistence loop) ---
    using FlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using GemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                        FlatmmConfig::kPadN,
                                                        FlatmmConfig::kPadK,
                                                        FlatmmConfig::DoubleSmemBuffer,
                                                        ALayout,
                                                        BLayout,
                                                        CLayout,
                                                        FlatmmConfig::TransposeC,
                                                        FlatmmConfig::UseStructuredSparsity,
                                                        /*Persistent=*/true,
                                                        FlatmmConfig::NumWaveGroups,
                                                        /*UseAsyncCopy=*/true>;

    // (HasHotLoop, TailNum) here are not load-bearing -- the MX pipeline
    // dispatches at runtime inside Run_ based on num_loop.
    using MXPipelineProblem =
        ck_tile::MXFlatmmPipelineProblem<ADataType,
                                         BDataType,
                                         AccDataType,
                                         FlatmmShape,
                                         GemmTraits,
                                         ck_tile::GemmPipelineScheduler::Default,
                                         /*HasHotLoop=*/true,
                                         ck_tile::TailNumber::Full>;

    using MXFlatmmPipeline =
        typename MXFlatmmArchTraits::template MXFlatmmPipeline<MXPipelineProblem>;

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
                                         FlatmmConfig::M_Warp,
                                         FlatmmConfig::N_Warp,
                                         FlatmmConfig::M_Warp_Tile,
                                         FlatmmConfig::N_Warp_Tile,
                                         FlatmmConfig::K_Warp_Tile,
                                         FlatmmConfig::TransposeC,
                                         FlatmmConfig::NumWaveGroups,
                                         false,
                                         1,
                                         MXFlatmmArchTraits::BlockedXDLN_PerWarp,
                                         FlatmmConfig::DoubleSmemBuffer>>;

    using Kernel = ck_tile::MXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;

    auto kargs        = Kernel::MakeKernelArgs(args);
    const dim3 grids  = Kernel::GridSize(kargs);
    const dim3 blocks = Kernel::BlockSize();

    const ck_tile::index_t total_tiles = (M / FlatmmConfig::M_Tile) * (N / FlatmmConfig::N_Tile);

    std::cout << "Launching persistent MXFlatmmKernel: " << Kernel::GetName() << "\n  grid: {"
              << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {" << blocks.x
              << "}" << "\n  M=" << M << ", N=" << N << ", K=" << K
              << ", total_tiles=" << total_tiles
              << ", multi_tile_per_block=" << (total_tiles > static_cast<int>(grids.x))
              << ", init_method=" << init_method << std::endl;

    // Guard the multi-tile-per-workgroup trigger: GridSize scales with the GPU,
    // so on a larger device the "multi-tile" cases can silently degrade into
    // trivial cases.
    if(expect_multi_tile)
        ASSERT_GT(total_tiles, static_cast<int>(grids.x))
            << "Test expected to exercise the multi-tile-per-workgroup path, but grid covers all "
               "tiles";

    auto s          = ck_tile::stream_config{nullptr, false, 0, 0, 1};
    ck_tile::ignore = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    c_dev.FromDevice(c_host.data());

    // --- CPU reference (the correctness oracle for both init regimes) ---
    ck_tile::HostTensor<CDataType> c_ref(
        ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<c_row_major>{}));
    c_ref.SetZero();

    ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, ScaleType, AccDataType, CDataType>(
        a_host, b_origin_host, c_ref, scale_a, scale_b);

    // Constant init (init_method==1) produces an exact integer K result; use
    // near-exact tolerance so a dropped/double-counted K-tile cannot hide inside
    // the K-scaled relative slack. Random init keeps 1e-2.
    const float rtol = (init_method == 1) ? 0.f : 1e-2f;
    const float atol = (init_method == 1) ? 1.f : 1e-2f;
    EXPECT_TRUE(
        ck_tile::check_err(c_host, c_ref, "MX persistent flatmm result mismatch", rtol, atol));
}

} // namespace

// ---- Sanity controls: single-tile, so the multi-tile path is not exercised;
//      these pass even with the bug present. ----

TEST(MXFlatmmPersistent, Single_Tile_Sanity_Random)
{
    run_persistent_test(
        /*M=*/128, /*N=*/256, /*K=*/256, /*init_method=*/0, /*expect_multi_tile=*/false);
}

// TODO: total_tiles must exceed the persistent grid size.
//       Dimensions are arch-conditional: dimensions must be large enough to
//       exceed the persistent grid size for the architecture. ----

constexpr ck_tile::index_t kMultiTileM = 2048;
constexpr ck_tile::index_t kMultiTileN = 8192;
constexpr ck_tile::index_t kMultiTileK = 1024;

TEST(MXFlatmmPersistent, Multi_Tile_Per_Block_Random)
{
    run_persistent_test(kMultiTileM,
                        kMultiTileN,
                        kMultiTileK,
                        /*init_method=*/0,
                        /*expect_multi_tile=*/true);
}
