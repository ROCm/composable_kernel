// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Multi-ABD GEMM with the gfx1250 producer/consumer TDM pipeline
// (GemmPipelineAgBgCrCompTDMProducerConsumer) and CShuffleEpilogue: E = CDE(A * B, D0, D1).
// The pipeline moves one A and one B tensor to LDS unmodified, so it takes exactly one A and one
// B with PassThrough element-wise functions; the D tensors are fused by the epilogue.

#include <algorithm>
#include <tuple>

#include "gtest/gtest.h"

#include "ck_tile/host.hpp"
#include "test_gemm_multi_abd_util.hpp"

namespace {

using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
class TestCkTileGemmMultiABDPC : public ::testing::Test
{
    protected:
    using ALayout         = std::tuple_element_t<0, Tuple>;
    using BLayout         = std::tuple_element_t<1, Tuple>;
    using ABDataType      = std::tuple_element_t<2, Tuple>;
    using DDataType       = std::tuple_element_t<3, Tuple>;
    using EDataType       = std::tuple_element_t<4, Tuple>;
    using CDElementWiseFn = std::tuple_element_t<5, Tuple>;
    using AccDataType     = F32;
    using DLayout         = Row;
    using ELayout         = Row;

    using AsLayout   = ck_tile::tuple<ALayout>;
    using AsDataType = ck_tile::tuple<ABDataType>;
    using BsLayout   = ck_tile::tuple<BLayout>;
    using BsDataType = ck_tile::tuple<ABDataType>;
    using DsLayout   = ck_tile::tuple<DLayout, DLayout>;
    using DsDataType = ck_tile::tuple<DDataType, DDataType>;

    static constexpr ck_tile::index_t M_Tile      = 128;
    static constexpr ck_tile::index_t N_Tile      = 128;
    static constexpr ck_tile::index_t K_Tile      = 64;
    static constexpr ck_tile::index_t M_Warp      = 2;
    static constexpr ck_tile::index_t N_Warp      = 2;
    static constexpr ck_tile::index_t K_Warp      = 1;
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<ABDataType, M_Warp_Tile>();

    // The TDM pipelines are unpadded and double-buffered; TransposeC follows the tile engine's
    // comp_tdm_pc setting.
    static constexpr bool DoubleSmemBuffer = true;
    static constexpr bool TransposeC       = false;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using Traits = ck_tile::TileGemmUniversalTraits<false, // kPadM
                                                    false, // kPadN
                                                    false, // kPadK
                                                    DoubleSmemBuffer,
                                                    AsLayout,
                                                    BsLayout,
                                                    ELayout,
                                                    TransposeC>;

    using Problem =
        ck_tile::UniversalGemmPipelineProblem<AsDataType,
                                              BsDataType,
                                              AccDataType,
                                              GemmShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;

    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompTDMProducerConsumer<Problem>;

    using GemmEpilogue =
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<AsDataType,
                                                                   BsDataType,
                                                                   DsDataType,
                                                                   AccDataType,
                                                                   EDataType,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   CDElementWiseFn,
                                                                   TilePartitioner::MPerBlock,
                                                                   TilePartitioner::NPerBlock,
                                                                   M_Warp,
                                                                   N_Warp,
                                                                   M_Warp_Tile,
                                                                   N_Warp_Tile,
                                                                   K_Warp_Tile,
                                                                   Problem::TransposeC,
                                                                   1,     // kNumWaveGroups
                                                                   false, // FixedVectorSize
                                                                   1,     // VectorSizeC
                                                                   1,     // BlockedXDLN_PerWarp
                                                                   DoubleSmemBuffer>>;

    using Kernel   = ck_tile::GemmKernelMultiABD<TilePartitioner, GemmPipeline, GemmEpilogue>;
    using HostArgs = ck_tile::GemmMultiABDHostArgs<1, 1, 2>;

    static auto Descriptor(std::size_t rows, std::size_t cols, std::size_t stride, auto layout)
    {
        using namespace ck_tile::literals;
        if constexpr(std::is_same_v<decltype(layout), Row>)
            return ck_tile::HostTensorDescriptor({rows, cols}, {stride, 1_uz});
        else
            return ck_tile::HostTensorDescriptor({rows, cols}, {1_uz, stride});
    }

    static int PackedStride(int rows, int cols, auto layout)
    {
        return std::is_same_v<decltype(layout), Row> ? cols : rows;
    }

    // Host-only: valid pointers are never dereferenced by IsSupportedArgument.
    static bool Supported(int M, int N, int K, int k_batch)
    {
        static float storage[64];
        const void* p = storage;
        HostArgs args({p},
                      {p},
                      {p, p},
                      storage,
                      k_batch,
                      M,
                      N,
                      K,
                      {PackedStride(M, K, ALayout{})},
                      {PackedStride(K, N, BLayout{})},
                      {N, N},
                      N);
        return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
    }

    void Run(int M, int N, int K)
    {
        if(!ck_tile::is_gfx125_supported())
        {
            GTEST_SKIP() << "the producer/consumer pipeline requires a gfx125 device";
        }
        const int stride_a = PackedStride(M, K, ALayout{});
        const int stride_b = PackedStride(K, N, BLayout{});

        ck_tile::HostTensor<ABDataType> a(Descriptor(M, K, stride_a, ALayout{}));
        ck_tile::HostTensor<ABDataType> b(Descriptor(K, N, stride_b, BLayout{}));
        ck_tile::HostTensor<DDataType> d0(Descriptor(M, N, N, DLayout{}));
        ck_tile::HostTensor<DDataType> d1(Descriptor(M, N, N, DLayout{}));
        ck_tile::HostTensor<EDataType> e(Descriptor(M, N, N, ELayout{}));

        ck_tile::FillUniformDistribution<ABDataType>{-1.f, 1.f}(a);
        ck_tile::FillUniformDistribution<ABDataType>{-1.f, 1.f}(b);
        ck_tile::FillUniformDistribution<DDataType>{-1.f, 1.f}(d0);
        ck_tile::FillUniformDistribution<DDataType>{-1.f, 1.f}(d1);

        ck_tile::DeviceMem a_buf(a.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_buf(b.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_buf(d0.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d1_buf(d1.get_element_space_size_in_bytes());
        ck_tile::DeviceMem e_buf(e.get_element_space_size_in_bytes());
        a_buf.ToDevice(a.data());
        b_buf.ToDevice(b.data());
        d0_buf.ToDevice(d0.data());
        d1_buf.ToDevice(d1.data());
        e_buf.SetZero();

        HostArgs args({a_buf.GetDeviceBuffer()},
                      {b_buf.GetDeviceBuffer()},
                      {d0_buf.GetDeviceBuffer(), d1_buf.GetDeviceBuffer()},
                      e_buf.GetDeviceBuffer(),
                      1,
                      M,
                      N,
                      K,
                      {stride_a},
                      {stride_b},
                      {N, N},
                      N);
        const auto kargs = Kernel::MakeKernelArgs(args);
        ASSERT_TRUE(Kernel::IsSupportedArgument(kargs));

        const dim3 grids  = Kernel::GridSize(M, N, 1);
        const dim3 blocks = Kernel::BlockSize();
        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false},
                               ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
        e_buf.FromDevice(e.data());

        ck_tile::HostTensor<ABDataType> a_ref(Descriptor(M, K, stride_a, ALayout{}));
        ck_tile::HostTensor<ABDataType> b_ref(Descriptor(K, N, stride_b, BLayout{}));
        ck_tile::HostTensor<EDataType> e_ref(Descriptor(M, N, N, ELayout{}));
        e_ref.SetZero();
        ck_tile::reference_gemm_multiple_abd<AsDataType,
                                             BsDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             ck_tile::element_wise::PassThrough,
                                             ck_tile::element_wise::PassThrough,
                                             CDElementWiseFn>(
            {a}, {b}, {d0, d1}, a_ref, b_ref, e_ref);

        const float max_value = *std::max_element(e_ref.mData.begin(), e_ref.mData.end());
        const auto tol =
            calculate_rtol_atol<ABDataType, ABDataType, AccDataType, EDataType, DDataType>(
                K, 1, max_value);
        EXPECT_TRUE(ck_tile::check_err(e,
                                       e_ref,
                                       "Error: Incorrect results!",
                                       tol.at(ck_tile::number<0>{}),
                                       tol.at(ck_tile::number<1>{})));
    }
};

// clang-format off
using KernelTypes = ::testing::Types<
    //         ALayout, BLayout, A/BDataType, DDataType, EDataType, CDElementWiseFn
    std::tuple<    Row,     Col,         F16,       F16,       F16, ElementWiseAddAdd>,
    std::tuple<    Row,     Col,         F16,      BF16,       F32, ElementWiseAddAdd>,
    std::tuple<    Row,     Row,         F16,       F16,       F16, MultiplyMultiply>,
    std::tuple<    Col,     Col,        BF16,      BF16,      BF16, ElementWiseAddAdd>
    >;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmMultiABDPC, KernelTypes);

TYPED_TEST(TestCkTileGemmMultiABDPC, Square_512x512x256) { this->Run(512, 512, 256); }

TYPED_TEST(TestCkTileGemmMultiABDPC, Rect_256x768x512) { this->Run(256, 768, 512); }

// M (the non-contiguous dimension of a row-major E) may be ragged: the C view masks it.
TYPED_TEST(TestCkTileGemmMultiABDPC, RaggedM_300x256x256) { this->Run(300, 256, 256); }

TYPED_TEST(TestCkTileGemmMultiABDPC, HostChecks)
{
    EXPECT_TRUE(TestFixture::Supported(256, 256, 256, 1));
    // A ragged contiguous dimension of E would write past the row end.
    EXPECT_FALSE(TestFixture::Supported(256, 250, 256, 1));
    // Multi-ABD rejects split-K.
    EXPECT_FALSE(TestFixture::Supported(256, 256, 256, 2));
}

} // namespace
