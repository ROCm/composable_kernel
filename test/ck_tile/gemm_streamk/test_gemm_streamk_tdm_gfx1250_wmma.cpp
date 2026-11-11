// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Stream-K on the gfx1250 TDM pipelines (comp_tdm and comp_tdm_v2). The CShuffle epilogue serves
// every reduction strategy; the TDM epilogue overwrites C, so it pairs with Linear and Tree only.

#include <algorithm>
#include <cstdint>
#include <utility>

#include "test_gemm_streamk_types.hpp"

struct CompTDMV1
{
    template <typename Problem>
    using pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV1<Problem>;
};

struct CompTDMV2
{
    template <typename Problem>
    using pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;
};

// Epilogues
using CShuffleStore = std::false_type;
using TdmStore      = std::true_type;

// Wave layouts (M x N x K waves); comp_tdm_v2 needs exactly four.
using Waves2x2 = ck_tile::sequence<2, 2, 1>;
using Waves4x2 = ck_tile::sequence<4, 2, 1>;
using Waves2x4 = ck_tile::sequence<2, 4, 1>;

template <typename Pipeline,
          typename Input,
          typename Reduction,
          typename Epilogue,
          typename Persistence,
          ck_tile::index_t Tile = 128,
          typename ALayout      = Row,
          typename BLayout      = Col,
          typename Waves        = Waves2x2>
using TdmStreamKParams = std::tuple<Pipeline,
                                    Input,
                                    Reduction,
                                    Epilogue,
                                    Persistence,
                                    ck_tile::number<Tile>,
                                    ALayout,
                                    BLayout,
                                    Waves>;

// comp_tdm on eight waves, with row-major A and column-major B.
template <typename Reduction, typename Epilogue, typename Persistence, typename Waves>
using TdmStreamKEightWaves =
    TdmStreamKParams<CompTDMV1, BF16, Reduction, Epilogue, Persistence, 128, Row, Col, Waves>;

#if GFX1250_STREAMK_SMOKE
// The gfx1250 emulator runs one workgroup at a time, so a Linear or Tree tile owner would wait
// forever for later workgroups. Cover both pipelines and persistence modes with atomic reduction.
using Gfx1250StreamKTdmTypes =
    ::testing::Types<TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, NonPersistent>,
                     TdmStreamKParams<CompTDMV2, BF16, Atomic, CShuffleStore, Persistent>,
                     TdmStreamKParams<CompTDMV2, F16, Atomic, CShuffleStore, NonPersistent>>;
#else
using Gfx1250StreamKTdmTypes = ::testing::Types<
    TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, TdmStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Tree, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Tree, TdmStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV1, BF16, Tree, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV1, BF16, Tree, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Atomic, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, TdmStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, BF16, Atomic, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, CShuffleStore, Persistent>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV2, F16, Atomic, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV2, F16, Linear, TdmStore, Persistent>,
    TdmStreamKParams<CompTDMV2, F16, Tree, CShuffleStore, NonPersistent>,
    TdmStreamKParams<CompTDMV1, BF16, Linear, TdmStore, NonPersistent, 64>,
    TdmStreamKParams<CompTDMV2, BF16, Atomic, CShuffleStore, Persistent, 64>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, NonPersistent, 64>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, NonPersistent, 128, Col, Row>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, CShuffleStore, Persistent, 128, Col, Row>,
    TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, NonPersistent, 128, Row, Row>,
    TdmStreamKParams<CompTDMV2, BF16, Linear, TdmStore, NonPersistent, 128, Row, Row>,
    TdmStreamKParams<CompTDMV2, BF16, Tree, TdmStore, NonPersistent, 128, Col, Col>,
    TdmStreamKParams<CompTDMV1, BF16, Atomic, CShuffleStore, Persistent, 128, Col, Col>,
    TdmStreamKEightWaves<Linear, CShuffleStore, NonPersistent, Waves4x2>,
    TdmStreamKEightWaves<Tree, TdmStore, Persistent, Waves4x2>,
    TdmStreamKEightWaves<Atomic, CShuffleStore, Persistent, Waves2x4>,
    TdmStreamKEightWaves<Linear, TdmStore, NonPersistent, Waves2x4>>;
#endif

template <typename Params>
class TestGfx1250StreamKTdm : public ::testing::Test
{
    protected:
    using Pipeline                          = std::tuple_element_t<0, Params>;
    using ADataType                         = std::tuple_element_t<1, Params>;
    using BDataType                         = ADataType;
    using AccDataType                       = F32;
    using CDataType                         = ADataType;
    static constexpr auto ReductionStrategy = std::tuple_element_t<2, Params>::value;
    static constexpr bool UseTdmEpilogue    = std::tuple_element_t<3, Params>::value;
    static constexpr bool Persistent        = std::tuple_element_t<4, Params>::value;
    static constexpr ck_tile::index_t Tile  = std::tuple_element_t<5, Params>::value;
    using ALayout                           = std::tuple_element_t<6, Params>;
    using BLayout                           = std::tuple_element_t<7, Params>;
    using CLayout                           = Row;
    using Waves                             = std::tuple_element_t<8, Params>;

    static constexpr ck_tile::index_t K_Tile = 64;

    using GemmShape = ck_tile::
        TileGemmShape<ck_tile::sequence<Tile, Tile, K_Tile>, Waves, ck_tile::sequence<16, 16, 32>>;
    using TilePartitioner =
        ck_tile::StreamKTilePartitioner<GemmShape, ReductionStrategy, Persistent>;
    // TDM clips every A and B load to the real extents, so the pipelines run unpadded.
    using Traits = ck_tile::TileGemmUniversalTraits<false,
                                                    false,
                                                    false,
                                                    /*DoubleSmemBuffer*/ true,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    false,
                                                    false,
                                                    Persistent>;
    using Problem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                              BDataType,
                                              AccDataType,
                                              GemmShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             ck_tile::tuple<>,
                                                             AccDataType,
                                                             CDataType,
                                                             ck_tile::tuple<>,
                                                             CLayout,
                                                             ck_tile::element_wise::PassThrough,
                                                             Tile,
                                                             Tile,
                                                             Waves::at(ck_tile::number<0>{}),
                                                             Waves::at(ck_tile::number<1>{}),
                                                             16,
                                                             16,
                                                             32,
                                                             Problem::TransposeC,
                                                             1,
                                                             false,
                                                             1,
                                                             1,
                                                             /*DoubleSmemBuffer*/ true>;
    using Epilogue        = std::conditional_t<UseTdmEpilogue,
                                               ck_tile::TdmEpilogue<EpilogueProblem>,
                                               ck_tile::CShuffleEpilogue<EpilogueProblem>>;
    using Kernel          = ck_tile::
        StreamKKernel<TilePartitioner, typename Pipeline::template pipeline<Problem>, Epilogue>;

    template <typename Layout>
    static ck_tile::index_t DefaultStride(ck_tile::index_t rows, ck_tile::index_t cols)
    {
        return std::is_same_v<Layout, Row> ? cols : rows;
    }

    // num_wgs is passed as num_cu with occupancy one: a fixed workgroup budget, so the partition
    // does not depend on the device. num_wgs = 0 takes the device's budget (CU count x occupancy).
    static auto MakeKargs(ck_tile::index_t M,
                          ck_tile::index_t N,
                          ck_tile::index_t K,
                          int num_wgs,
                          const void* a = nullptr,
                          const void* b = nullptr,
                          void* c       = nullptr)
    {
        const ck_tile::StreamKHostArgs args{a,
                                            b,
                                            c,
                                            M,
                                            N,
                                            K,
                                            DefaultStride<ALayout>(M, K),
                                            DefaultStride<BLayout>(K, N),
                                            DefaultStride<CLayout>(M, N)};
        return num_wgs > 0 ? Kernel::MakeKernelArgs(args, num_wgs, 1)
                           : Kernel::MakeKernelArgs(args);
    }

    // Stream-K workgroups that store a partial tile (their first iteration is not a tile's
    // first) and that run more than one tile segment.
    static std::pair<int, int> CountSplitSegments(const TilePartitioner& p)
    {
        int partial = 0;
        int multi   = 0;
        for(int cta = 0; cta < p.get_sk_ctas(); ++cta)
        {
            const int start = cta * p.get_iters_per_sk_cta() + std::min(cta, p.get_extra_iters());
            const int end   = start + p.get_iters_per_sk_cta() + (cta < p.get_extra_iters());
            partial += start % p.get_iters_per_tile() != 0;
            multi +=
                end > start && start / p.get_iters_per_tile() != (end - 1) / p.get_iters_per_tile();
        }
        return {partial, multi};
    }

    public:
    enum struct Split
    {
        None,      // data-parallel tiles only
        Partial,   // some workgroup stores a partial tile
        MultiTile, // and some workgroup runs segments of two tiles back to back
        Any
    };

    // Runs M x N x K on num_wgs workgroups and checks C, or checks that IsSupportedArgument
    // rejects the problem when supported is false.
    void Run(ck_tile::index_t M,
             ck_tile::index_t N,
             ck_tile::index_t K,
             int num_wgs,
             Split split    = Split::Any,
             bool supported = true)
    {
        SCOPED_TRACE(::testing::Message()
                     << "M: " << M << ", N: " << N << ", K: " << K << ", workgroups: " << num_wgs);

        const ck_tile::index_t stride_A = DefaultStride<ALayout>(M, K);
        const ck_tile::index_t stride_B = DefaultStride<BLayout>(K, N);
        const ck_tile::index_t stride_C = DefaultStride<CLayout>(M, N);
        const auto f_desc               = [](std::size_t rows, std::size_t cols, auto layout) {
            const std::size_t stride = DefaultStride<decltype(layout)>(rows, cols);
            if constexpr(std::is_same_v<decltype(layout), Row>)
                return ck_tile::HostTensorDescriptor({rows, cols}, {stride, std::size_t{1}});
            else
                return ck_tile::HostTensorDescriptor({rows, cols}, {std::size_t{1}, stride});
        };

        ck_tile::HostTensor<ADataType> a_m_k(f_desc(M, K, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n(f_desc(K, N, BLayout{}));
        ck_tile::HostTensor<CDataType> c_m_n(f_desc(M, N, CLayout{}));
        ck_tile::HostTensor<CDataType> c_m_n_ref(f_desc(M, N, CLayout{}));
        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, /*seed*/ 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, /*seed*/ 11940}(b_k_n);

        ck_tile::DeviceMem a_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_buf(c_m_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_ref_buf(c_m_n_ref.get_element_space_size_in_bytes());
        a_buf.ToDevice(a_m_k.data());
        b_buf.ToDevice(b_k_n.data());
        c_buf.SetZero();
        c_ref_buf.SetZero();

        auto kargs = MakeKargs(M,
                               N,
                               K,
                               num_wgs,
                               a_buf.GetDeviceBuffer(),
                               b_buf.GetDeviceBuffer(),
                               c_buf.GetDeviceBuffer());
        ASSERT_EQ(Kernel::IsSupportedArgument(kargs), supported);
        if(!supported)
        {
            return;
        }

        const auto& partitioner     = kargs.tile_partitioner;
        const auto [partial, multi] = CountSplitSegments(partitioner);
        if(split == Split::None)
        {
            EXPECT_EQ(partitioner.get_sk_tiles(), 0);
        }
        // A data-parallel launch cannot validate cross-workgroup reduction.
        if(split == Split::Partial || split == Split::MultiTile)
        {
            EXPECT_GT(partial, 0);
        }
        if(split == Split::MultiTile)
        {
            EXPECT_GT(multi, 0);
        }

        ck_tile::DeviceMem workspace(Kernel::GetWorkSpaceSize(kargs));
        workspace.SetZero();
        kargs.workspace_ptr = workspace.GetDeviceBuffer();
        ck_tile::ignore     = ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0, 0, 1},
            ck_tile::make_kernel<1>(
                Kernel{}, Kernel::GridSize(partitioner), Kernel::BlockSize(), 0, kargs));
        c_buf.FromDevice(c_m_n.data());

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                                             static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                                             static_cast<CDataType*>(c_ref_buf.GetDeviceBuffer()),
                                             M,
                                             N,
                                             K,
                                             stride_A,
                                             stride_B,
                                             stride_C);
        c_ref_buf.FromDevice(c_m_n_ref.data());

        const float max_accumulated_value =
            *std::max_element(c_m_n_ref.mData.begin(), c_m_n_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, partitioner.estimate_num_wgs_per_tile(), max_accumulated_value);
        EXPECT_TRUE(ck_tile::check_err(c_m_n,
                                       c_m_n_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{})));
    }
};

TYPED_TEST_SUITE(TestGfx1250StreamKTdm, Gfx1250StreamKTdmTypes);

TYPED_TEST(TestGfx1250StreamKTdm, DataParallelOnly)
{
    constexpr auto T = TestFixture::Tile;
    this->Run(2 * T, 2 * T, 512, 4, TestFixture::Split::None);
}

TYPED_TEST(TestGfx1250StreamKTdm, SplitKSegments)
{
    // Eight workgroups share one tile with 1 to 5 K iterations each: both tails of the double
    // LDS buffer, with and without the hot loop.
    constexpr auto T = TestFixture::Tile;
    for(int loops : {1, 2, 3, 4, 5})
        this->Run(T, T, 8 * 64 * loops, 8, TestFixture::Split::Partial);
}

TYPED_TEST(TestGfx1250StreamKTdm, SegmentsAcrossTiles)
{
    // Workgroups that finish one tile with a short (odd) segment and start the next tile in the
    // same launch: the next segment refills the LDS that the clipped prefetch targets.
    constexpr auto T = TestFixture::Tile;
    this->Run(T, 3 * T, 3 * 64, 4, TestFixture::Split::MultiTile);
    this->Run(T, 2 * T, 5 * 64, 4, TestFixture::Split::MultiTile);
}

TYPED_TEST(TestGfx1250StreamKTdm, MixedDataParallelAndStreamK)
{
    // 33 tiles on eight workgroups: data-parallel tiles, then Stream-K segments that span two
    // tiles. Persistent kernels also reuse LDS across consecutive data-parallel tiles.
    constexpr auto T = TestFixture::Tile;
    this->Run(T, 33 * T, 512, 8, TestFixture::Split::MultiTile);
}

TYPED_TEST(TestGfx1250StreamKTdm, RaggedK)
{
    // The last K tile of a segment is clipped by TDM.
    constexpr auto T = TestFixture::Tile;
    this->Run(T, 3 * T, 1000, 8, TestFixture::Split::Partial);
    this->Run(T, 3 * T, 544, 5, TestFixture::Split::Partial);
}

TYPED_TEST(TestGfx1250StreamKTdm, RaggedMN)
{
    constexpr auto T = TestFixture::Tile;
    // A ragged M tile lies past the last row of C, which both epilogues drop.
    this->Run(T + 8, 2 * T, 1024, 8);
    // N is the contiguous dimension of C: the TDM epilogue clips a ragged N tile, but CShuffle
    // would store it into the next row, so IsSupportedArgument rejects it.
    this->Run(T, 2 * T + 8, 1024, 8, TestFixture::Split::Any, TestFixture::UseTdmEpilogue);
    this->Run(118, 102, 8 * 64 * 3, 8, TestFixture::Split::Any, TestFixture::UseTdmEpilogue);
}

#if !GFX1250_STREAMK_SMOKE
TYPED_TEST(TestGfx1250StreamKTdm, FullDevice)
{
    // The device's workgroup budget: a Linear or Tree workgroup waits for later ones, so every
    // Stream-K workgroup the occupancy query allows must be resident at once.
    constexpr auto T = TestFixture::Tile;
    const auto kargs = TestFixture::MakeKargs(4 * T, 4 * T, 64 * 255, 0);
    EXPECT_EQ(kargs.tile_partitioner.get_sk_ctas(), kargs.tile_partitioner.get_max_active_wgs());
    this->Run(4 * T, 4 * T, 64 * 255, 0, TestFixture::Split::Partial);
}
#endif

TYPED_TEST(TestGfx1250StreamKTdm, RejectsMisalignedPointers)
{
    // TDM addresses A, B and C in whole elements.
    constexpr auto T      = TestFixture::Tile;
    const auto misaligned = reinterpret_cast<void*>(std::uintptr_t{1});
    EXPECT_TRUE(TestFixture::Kernel::IsSupportedArgument(TestFixture::MakeKargs(T, T, 512, 8)));
    EXPECT_FALSE(TestFixture::Kernel::IsSupportedArgument(
        TestFixture::MakeKargs(T, T, 512, 8, misaligned, nullptr, nullptr)));
    EXPECT_FALSE(TestFixture::Kernel::IsSupportedArgument(
        TestFixture::MakeKargs(T, T, 512, 8, nullptr, nullptr, misaligned)));
}
