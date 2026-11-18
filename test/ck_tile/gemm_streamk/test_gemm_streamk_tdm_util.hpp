// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Shared harness for the gfx1250 TDM Stream-K tests (comp_tdm, comp_tdm_v2, and comp_tdm_v3).

#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "test_gemm_streamk_types.hpp"

// Epilogues
using CShuffleStore = std::false_type;
using TdmStore      = std::true_type;

// M, N, and K padding flags
using Unpadded = std::false_type;
using Padded   = std::true_type;

template <template <typename> class PipelineT,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename BlockTile_,
          typename BlockWarps_,
          typename WarpTile_,
          typename Reduction,
          typename Persistence,
          typename Epilogue = CShuffleStore,
          typename Padding  = Unpadded>
struct StreamKTdmConfig
{
    template <typename Problem>
    using Pipeline = PipelineT<Problem>;

    using ALayout     = ALayout_;
    using BLayout     = BLayout_;
    using CLayout     = CLayout_;
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using AccDataType = AccDataType_;
    using CDataType   = CDataType_;
    using BlockTile   = BlockTile_;
    using BlockWarps  = BlockWarps_;
    using WarpTile    = WarpTile_;

    static constexpr auto ReductionStrategy = Reduction::value;
    static constexpr bool Persistent        = Persistence::value;
    static constexpr bool UseTdmEpilogue    = Epilogue::value;
    static constexpr bool Pad               = Padding::value;
};

enum struct StreamKTdmSplit
{
    None,      // data-parallel tiles only
    FullTile,  // Stream-K active, full tiles only (no cross-workgroup partial reduction)
    Partial,   // some workgroup stores a partial tile
    MultiTile, // and some workgroup runs segments of two tiles back to back
    Any
};

template <typename Config>
struct StreamKTdmHarness
{
    using Split       = StreamKTdmSplit;
    using ADataType   = typename Config::ADataType;
    using BDataType   = typename Config::BDataType;
    using AccDataType = typename Config::AccDataType;
    using CDataType   = typename Config::CDataType;
    using ALayout     = typename Config::ALayout;
    using BLayout     = typename Config::BLayout;
    using CLayout     = typename Config::CLayout;
    using BlockTile   = typename Config::BlockTile;
    using BlockWarps  = typename Config::BlockWarps;
    using WarpTile    = typename Config::WarpTile;

    static constexpr bool UseTdmEpilogue    = Config::UseTdmEpilogue;
    static constexpr ck_tile::index_t TileM = BlockTile::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t TileN = BlockTile::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t TileK = BlockTile::at(ck_tile::number<2>{});

    using GemmShape = ck_tile::TileGemmShape<BlockTile, BlockWarps, WarpTile>;
    using TilePartitioner =
        ck_tile::StreamKTilePartitioner<GemmShape, Config::ReductionStrategy, Config::Persistent>;
    // TDM clips every A and B load to the real extents, so the pipelines also run unpadded.
    using Traits = ck_tile::TileGemmUniversalTraits<Config::Pad,
                                                    Config::Pad,
                                                    Config::Pad,
                                                    /*DoubleSmemBuffer*/ true,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    false,
                                                    false,
                                                    Config::Persistent>;
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
                                                             TileM,
                                                             TileN,
                                                             BlockWarps::at(ck_tile::number<0>{}),
                                                             BlockWarps::at(ck_tile::number<1>{}),
                                                             WarpTile::at(ck_tile::number<0>{}),
                                                             WarpTile::at(ck_tile::number<1>{}),
                                                             WarpTile::at(ck_tile::number<2>{}),
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
        StreamKKernel<TilePartitioner, typename Config::template Pipeline<Problem>, Epilogue>;

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

    // Runs M x N x K on num_wgs workgroups and checks C, or checks that IsSupportedArgument
    // rejects the problem when supported is false.
    static void Run(ck_tile::index_t M,
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
        if(split == Split::FullTile)
        {
            EXPECT_GT(partitioner.get_sk_tiles(), 0);
            EXPECT_EQ(partial, 0);
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
