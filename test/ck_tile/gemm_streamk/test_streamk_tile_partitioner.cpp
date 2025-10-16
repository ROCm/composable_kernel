// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "test_streamk_tile_partitioner_common.hpp"

TEST(StreamKTilePartitionerBaseConstructor, SKOnly)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerBaseExpected expected_values{
        2, 0, 3, 4, 1, 2, 1, 0, 2, Config::GRID, Config::N};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseConstructor, DPOnly)
{
    using Config = StreamKTilePartitionerBaseConfigDPOnly;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerBaseExpected expected_values{
        0, 6, 0, 0, 0, 2, 0, 12, 6, Config::GRID, Config::N};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseConstructor, DP2TileSK)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerBaseExpected expected_values{
        4, 3, 3, 8, 2, 2, 2, 6, 7, Config::GRID, Config::N};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseConstructor, EdgeCase)
{
    using Config = StreamKTilePartitionerBaseConfigEdgeCase;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerBaseExpected expected_values{
        0, 1, 0, 0, 0, 2, 0, 2, 1, Config::GRID, Config::N};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseGetWorkSpaceSize, AtomicStrategy)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};

    EXPECT_EQ(tile_partitioner.get_workspace_size(sizeof(float)), 0);
}

TEST(StreamKTilePartitionerBaseGetWorkSpaceSize, ReductionStrategy)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape,
                                        ck_tile::StreamKReductionStrategy::Reduction>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    ck_tile::index_t expected_partials_size =
        sizeof(float) * Config::M_TILE * Config::N_TILE * Config::GRID;
    ck_tile::index_t expected_flags_size = sizeof(ck_tile::index_t) * Config::GRID;

    EXPECT_EQ(tile_partitioner.get_workspace_size(sizeof(float)),
              expected_partials_size + expected_flags_size);
}

TEST(StreamKTilePartitionerBaseGetLocalIter, GetLocalIter)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigSKOnly;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel          = KernelWrapperSpecialized<TilePartitioner,
                                                     StreamKTilePartitionerBaseMethodId::GET_LOCAL_ITER>;

    // Test parameters
    ck_tile::DeviceMem local_iter_start_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t iter_start      = 3;
    ck_tile::index_t tile_iter_start = 2;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(iter_start,
                                        tile_iter_start,
                                        Config::UNUSED,
                                        local_iter_start_dev.GetDeviceBuffer(),
                                        nullptr,
                                        Config::UNUSED);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate result
    ck_tile::index_t local_iter_start;
    local_iter_start_dev.FromDevice(&local_iter_start);
    EXPECT_EQ(local_iter_start, iter_start - tile_iter_start);
}

TEST(StreamKTilePartitionerBaseGetLocalIterEnd, MinIsTileIterEnd)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigDP2TileSK;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel          = KernelWrapperSpecialized<TilePartitioner,
                                                     StreamKTilePartitionerBaseMethodId::GET_LOCAL_ITER_END>;
    // Test parameters
    ck_tile::DeviceMem local_iter_end_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t tile_iter_start = 6;
    ck_tile::index_t iter_end        = 9;
    ck_tile::index_t tile_iter_end   = 8;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(tile_iter_start,
                                        iter_end,
                                        tile_iter_end,
                                        local_iter_end_dev.GetDeviceBuffer(),
                                        nullptr,
                                        Config::UNUSED);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t local_iter_end;
    local_iter_end_dev.FromDevice(&local_iter_end);
    EXPECT_EQ(local_iter_end, tile_iter_end - tile_iter_start);
}

TEST(StreamKTilePartitionerBaseGetLocalIterEnd, MinIsIterEnd)
{
    // Types
    // Note: For this test, the Config is used for types only, the function get_local_iter_end is
    // static; thus, the test parameters are independent of the Config in this case.
    using Config          = StreamKTilePartitionerBaseConfigDP2TileSK;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel          = KernelWrapperSpecialized<TilePartitioner,
                                                     StreamKTilePartitionerBaseMethodId::GET_LOCAL_ITER_END>;
    // Test parameters
    ck_tile::DeviceMem local_iter_end_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t tile_iter_start = 12;
    ck_tile::index_t iter_end        = 13;
    ck_tile::index_t tile_iter_end   = 14;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(tile_iter_start,
                                        iter_end,
                                        tile_iter_end,
                                        local_iter_end_dev.GetDeviceBuffer(),
                                        nullptr,
                                        Config::UNUSED);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t local_iter_end;
    local_iter_end_dev.FromDevice(&local_iter_end);
    EXPECT_EQ(local_iter_end, iter_end - tile_iter_start);
}

TEST(StreamKTilePartitionerBaseGetTileBoundaries, GetTileBoundaries)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigSKOnly;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel =
        KernelWrapperSpecialized<TilePartitioner,
                                 StreamKTilePartitionerBaseMethodId::GET_TILE_BOUNDARIES>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};
    ck_tile::DeviceMem tile_iter_start_dev(sizeof(ck_tile::index_t));
    ck_tile::DeviceMem tile_iter_end_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t tile_idx = 1;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(Config::PLACEHOLDER,
                                        Config::PLACEHOLDER,
                                        tile_idx,
                                        tile_iter_start_dev.GetDeviceBuffer(),
                                        tile_iter_end_dev.GetDeviceBuffer(),
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t tile_iter_start, tile_iter_end;
    tile_iter_start_dev.FromDevice(&tile_iter_start);
    tile_iter_end_dev.FromDevice(&tile_iter_end);
    // There are 2 iters per tile. Thus, for tile_idx 1, we expect 2 and 4 to be the start and end,
    // respectively.
    EXPECT_EQ(tile_iter_start, 2);
    EXPECT_EQ(tile_iter_end, 4);
}

TEST(StreamKTilePartitionerBaseGetTileIndex, GetTileIndex)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigDP2TileSK;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel          = KernelWrapperSpecialized<TilePartitioner,
                                                     StreamKTilePartitionerBaseMethodId::GET_TILE_INDEX>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};
    ck_tile::DeviceMem tile_idx_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t iter_start = 8;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(iter_start,
                                        Config::UNUSED,
                                        Config::UNUSED,
                                        tile_idx_dev.GetDeviceBuffer(),
                                        nullptr,
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t tile_idx;
    tile_idx_dev.FromDevice(&tile_idx);
    // Since there are 2 iters per tile, iter 8 maps to tile_idx 4.
    EXPECT_EQ(tile_idx, 4);
}

TEST(StreamKTilePartitionerBaseGetIterBoundaries, ZeroExtraItersBeforeMe)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigDP2TileSK;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel =
        KernelWrapperSpecialized<TilePartitioner,
                                 StreamKTilePartitionerBaseMethodId::GET_ITER_BOUNDARIES>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};
    ck_tile::DeviceMem iter_start_dev(sizeof(ck_tile::index_t));
    ck_tile::DeviceMem iter_end_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t cta_idx = 0;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(Config::PLACEHOLDER,
                                        Config::PLACEHOLDER,
                                        cta_idx,
                                        iter_start_dev.GetDeviceBuffer(),
                                        iter_end_dev.GetDeviceBuffer(),
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t iter_start, iter_end;
    iter_start_dev.FromDevice(&iter_start);
    iter_end_dev.FromDevice(&iter_end);
    EXPECT_EQ(iter_start, 6);
    EXPECT_EQ(iter_end, 9);
}

TEST(StreamKTilePartitionerBaseGetIterBoundaries, NonZeroExtraItersBeforeMe)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigDP2TileSK;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel =
        KernelWrapperSpecialized<TilePartitioner,
                                 StreamKTilePartitionerBaseMethodId::GET_ITER_BOUNDARIES>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};
    ck_tile::DeviceMem iter_start_dev(sizeof(ck_tile::index_t));
    ck_tile::DeviceMem iter_end_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t cta_idx = 1;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(Config::PLACEHOLDER,
                                        Config::PLACEHOLDER,
                                        cta_idx,
                                        iter_start_dev.GetDeviceBuffer(),
                                        iter_end_dev.GetDeviceBuffer(),
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t iter_start, iter_end;
    iter_start_dev.FromDevice(&iter_start);
    iter_end_dev.FromDevice(&iter_end);
    EXPECT_EQ(iter_start, 9);
    EXPECT_EQ(iter_end, 12);
}

TEST(StreamKTilePartitionerBaseGetIterBoundaries, MinIsExtraIters)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigDP2TileSK;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel =
        KernelWrapperSpecialized<TilePartitioner,
                                 StreamKTilePartitionerBaseMethodId::GET_ITER_BOUNDARIES>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};
    ck_tile::DeviceMem iter_start_dev(sizeof(ck_tile::index_t));
    ck_tile::DeviceMem iter_end_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t cta_idx = 2;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(Config::PLACEHOLDER,
                                        Config::PLACEHOLDER,
                                        cta_idx,
                                        iter_start_dev.GetDeviceBuffer(),
                                        iter_end_dev.GetDeviceBuffer(),
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t iter_start, iter_end;
    iter_start_dev.FromDevice(&iter_start);
    iter_end_dev.FromDevice(&iter_end);
    EXPECT_EQ(iter_start, 12);
    EXPECT_EQ(iter_end, 14);
}

TEST(StreamKTilePartitionerBaseGetOutputTileIndex, TestAllMappings)
{
    using Config                   = StreamKTilePartitionerBaseConfigLargerCTensor;
    ck_tile::index_t m_macro_tiles = Config::M / Config::M_TILE;
    ck_tile::index_t n_macro_tiles = Config::N / Config::N_TILE;
    ck_tile::index_t tile_idx      = 0;

    for(ck_tile::index_t row = 0; row < m_macro_tiles; ++row)
    {
        for(ck_tile::index_t col = 0; col < n_macro_tiles; ++col)
        {
            test_get_output_tile_index(tile_idx, ck_tile::make_tuple(row, col));
            ++tile_idx;
        }
    }
}

// Persistent
TEST(StreamKTilePartitioner_v2_PersistentConstructor, SKOnly)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::StreamKTilePartitioner_v2<Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerV2PersistentExpected expected_values{0, 0, 3};
    validate_streamk_v2_persistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_v2_PersistentConstructor, DPOnly)
{
    using Config = StreamKTilePartitionerBaseConfigDPOnly;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerV2PersistentExpected expected_values{2, 0, 3};
    validate_streamk_v2_persistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_v2_PersistentConstructor, DP2TileSK)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerV2PersistentExpected expected_values{1, 0, 3};
    validate_streamk_v2_persistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_v2_PersistentConstructor, EdgeCase)
{
    using Config = StreamKTilePartitionerBaseConfigEdgeCase;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerV2PersistentExpected expected_values{0, 1, 4};
    validate_streamk_v2_persistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_v2_GridSize_Persistent, SKOnly)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    const auto g = tile_partitioner.grid_size();
    EXPECT_EQ(g.x, Config::GRID);
}

TEST(StreamKTilePartitioner_v2_GridSize_Persistent, EdgeCase)
{
    using Config = StreamKTilePartitionerBaseConfigEdgeCase;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    const auto g = tile_partitioner.grid_size();
    EXPECT_EQ(g.x, 1);
}

// Non-Persistent Tests
TEST(StreamKTilePartitioner_v2_NonPersistentConstructor, SKOnly)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::StreamKTilePartitioner_v2<Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerV2NonPersistentExpected expected_values{0, 0, 0, 3};
    validate_streamk_v2_nonpersistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_v2_NonPersistentConstructor, DPOnly)
{
    using Config = StreamKTilePartitionerBaseConfigDPOnly;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerV2NonPersistentExpected expected_values{6, 0, 6, 3};
    validate_streamk_v2_nonpersistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_v2_NonPersistentConstructor, DP2TileSK)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerV2NonPersistentExpected expected_values{3, 0, 3, 3};
    validate_streamk_v2_nonpersistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_v2_NonPersistentConstructor, EdgeCase)
{
    using Config = StreamKTilePartitionerBaseConfigEdgeCase;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    StreamKTilePartitionerV2NonPersistentExpected expected_values{1, 0, 1, 4};
    validate_streamk_v2_nonpersistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_v2_GridSize_NonPersistent, DP2TileSK)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitioner_v2<typename Config::GemmShape,
                                       ck_tile::StreamKReductionStrategy::Atomic,
                                       false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::GRID};

    const auto g = tile_partitioner.grid_size();
    EXPECT_EQ(g.x, 6);
}
