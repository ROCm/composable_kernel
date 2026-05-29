// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_streamk_tile_partitioner_common.hpp"
#include "ck_tile/host/device_prop.hpp"

TEST(StreamKTilePartitionerBaseConstructor, SKOnly)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerBaseExpected expected_values{
        2, 0, 3, 4, 1, 2, 1, 0, 2, Config::MAX_ACTIVE_WGS, Config::N, Config::K, 0};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseConstructor, DPOnly)
{
    using Config = StreamKTilePartitionerBaseConfigDPOnly;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerBaseExpected expected_values{
        0, 6, 0, 0, 0, 2, 0, 12, 6, Config::MAX_ACTIVE_WGS, Config::N, Config::K, 0};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseConstructor, DP2TileSK)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerBaseExpected expected_values{
        4, 3, 3, 8, 2, 2, 2, 6, 7, Config::MAX_ACTIVE_WGS, Config::N, Config::K, 0};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseConstructor, EdgeCase)
{
    using Config = StreamKTilePartitionerBaseConfigEdgeCase;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerBaseExpected expected_values{
        0, 1, 0, 0, 0, 2, 0, 2, 1, Config::MAX_ACTIVE_WGS, Config::N, Config::K, 0};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseConstructor, RemainderAlongK)
{
    using Config = StreamKTilePartitionerBaseConfigRemainderAlongK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerBaseExpected expected_values{
        1, 0, 2, 3, 1, 3, 1, 0, 1, Config::MAX_ACTIVE_WGS, Config::N, Config::K, 1};
    validate_streamk_base_constructor<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitionerBaseGetFlagsBufferSize, FlagsLessThan128Bytes)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape,
                                        ck_tile::StreamKReductionStrategy::Linear>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    EXPECT_EQ(tile_partitioner.get_flags_buffer_size(), 128);
}

TEST(StreamKTilePartitionerBaseGetFlagsBufferSize, FlagsEqual128Bytes)
{
    using Config = StreamKTilePartitionerBaseConfigFlagsSizeEqual128Bytes;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape,
                                        ck_tile::StreamKReductionStrategy::Linear>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    EXPECT_EQ(tile_partitioner.get_flags_buffer_size(), 128);
}

TEST(StreamKTilePartitionerBaseGetFlagsBufferSize, FlagsGreaterThan128Bytes)
{
    using Config = StreamKTilePartitionerBaseConfigFlagsSizeGreaterThan128Bytes;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape,
                                        ck_tile::StreamKReductionStrategy::Linear>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    EXPECT_EQ(tile_partitioner.get_flags_buffer_size(), 256);
}

TEST(StreamKTilePartitionerBaseGetWorkSpaceSize, AtomicStrategy)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    EXPECT_EQ(tile_partitioner.get_workspace_size(sizeof(float)), 0);
}

TEST(StreamKTilePartitionerBaseGetWorkSpaceSize, ReductionStrategy)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape,
                                        ck_tile::StreamKReductionStrategy::Linear>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    ck_tile::index_t expected_partials_size =
        sizeof(float) * Config::M_TILE * Config::N_TILE * Config::MAX_ACTIVE_WGS;
    // Since MAX_ACTIVE_WGS is 3, the final padded flags array must be 128B to ensure the total byte
    // size of the flags array is 128B-aligned.
    ck_tile::index_t expected_flags_size = 128;

    EXPECT_EQ(tile_partitioner.get_workspace_size(sizeof(float)),
              expected_partials_size + expected_flags_size);
}

TEST(StreamKTilePartitionerBaseEstimateNumWgsPerTile, EstimateNumWgsPerTileLowerValue)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    EXPECT_EQ(tile_partitioner.estimate_num_wgs_per_tile(), 2);
}

TEST(StreamKTilePartitionerBaseEstimateNumWgsPerTile, EstimateNumWgsPerTileEqualValue)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnlyWith2WgsPerSKTile;

    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    EXPECT_EQ(tile_partitioner.estimate_num_wgs_per_tile(), 2);
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
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};
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
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};
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
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};
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
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};
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
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};
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

TEST(StreamKTilePartitionerBaseRemapXCD, SmallArray)
{
    int num_xcds = 8;
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::
        StreamKTilePartitioner<Config::GemmShape, ck_tile::StreamKReductionStrategy::Atomic, true>
            tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    const std::vector<ck_tile::index_t> initial_values = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::vector<ck_tile::index_t> expected_values = {
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};

    test_remap_xcd<Config::GemmShape>(initial_values, expected_values, tile_partitioner, num_xcds);
}

TEST(StreamKTilePartitionerBaseRemapXCD, MidArray)
{
    int num_xcds = 8;
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::
        StreamKTilePartitioner<Config::GemmShape, ck_tile::StreamKReductionStrategy::Atomic, true>
            tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    const std::vector<ck_tile::index_t> initial_values = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
        64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
        96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126};
    const std::vector<ck_tile::index_t> expected_values = {
        0,  16, 32,  48,  64, 80, 96,  112, 1,  17, 33,  49,  65, 81, 97,  113, 2,  18, 34,  50,
        66, 82, 98,  114, 3,  19, 35,  51,  67, 83, 99,  115, 4,  20, 36,  52,  68, 84, 100, 116,
        5,  21, 37,  53,  69, 85, 101, 117, 6,  22, 38,  54,  70, 86, 102, 118, 7,  23, 39,  55,
        71, 87, 103, 119, 8,  24, 40,  56,  72, 88, 104, 120, 9,  25, 41,  57,  73, 89, 105, 121,
        10, 26, 42,  58,  74, 90, 106, 122, 11, 27, 43,  59,  75, 91, 107, 123, 12, 28, 44,  60,
        76, 92, 108, 124, 13, 29, 45,  61,  77, 93, 109, 125, 14, 30, 46,  62,  78, 94, 110, 126,
        15, 31, 47,  63,  79, 95, 111};
    test_remap_xcd<Config::GemmShape>(initial_values, expected_values, tile_partitioner, num_xcds);
}

TEST(StreamKTilePartitionerBaseRemapXCD, UnevenXCD)
{
    constexpr int num_xcds = 5;
    using Config           = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::
        StreamKTilePartitioner<Config::GemmShape, ck_tile::StreamKReductionStrategy::Atomic, true>
            tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    const std::vector<ck_tile::index_t> initial_values = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::vector<ck_tile::index_t> expected_values = {
        0, 4, 7, 10, 13, 1, 5, 8, 11, 14, 2, 6, 9, 12, 15, 3};

    test_remap_xcd<Config::GemmShape>(initial_values, expected_values, tile_partitioner, num_xcds);
}

TEST(StreamKTilePartitionerBaseRemapXCD, SingleXCD)
{
    constexpr int num_xcds = 1;
    using Config           = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::
        StreamKTilePartitioner<Config::GemmShape, ck_tile::StreamKReductionStrategy::Atomic, true>
            tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    const std::vector<ck_tile::index_t> initial_values = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::vector<ck_tile::index_t> expected_values = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    test_remap_xcd<Config::GemmShape>(initial_values, expected_values, tile_partitioner, num_xcds);
}

TEST(StreamKTilePartitionerBaseGetTileLocalCtaIndex, SKOnlyLargeK)
{
    /*
    The StreamKTilePartitionerBaseConfigSKOnlyLargeK has the following form:
    - tiles in the C tensor: 2
    - iters_per_tile: 5
    - grid: 5
    - dp_tiles: 0
    - sk_tiles: 2
    - iters_per_sk_cta: 2
    - extra_iters: 0

    The tiles with iters are as follows:

    tile_idx: __________0_________|_________1_________|
    tile_iter:| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
              |   |   |   |   |   |   |   |   |   |   |
              <---------------SK Tiles--------------->|

    From the above configuration, we get the following:
    - SK CTA 0: tile_iter_start is 0 with local CTA index of 0 in tile 0
    - SK CTA 1: tile_iter_start is 0 with local CTA index of 1 in tile 0
    - SK CTA 2: tile_iter_start is 0 with local CTA index of 2 in tile 0
    - SK CTA 2: tile_iter_start is 5 with local CTA index of 0 in tile 1
    - SK CTA 3: tile_iter_start is 5 with local CTA index of 1 in tile 1
    - SK CTA 4: tile_iter_start is 5 with local CTA index of 2 in tile 1
    */

    // Now we create a vector of triplets (tile_iter_start, cta_idx, tile_local_cta_idx) to test
    std::vector<std::array<ck_tile::index_t, 3>> sk_triplets{
        {0, 0, 0}, {0, 1, 1}, {0, 2, 2}, {5, 2, 0}, {5, 3, 1}, {5, 4, 2}};

    for(const auto& triplet : sk_triplets)
    {
        const auto& [tile_iter_start, cta_idx, tile_local_cta_idx] = triplet;
        test_get_tile_local_cta_idx<StreamKTilePartitionerBaseConfigSKOnlyLargeK>(
            tile_iter_start, cta_idx, tile_local_cta_idx);
    }
}

TEST(StreamKTilePartitionerBaseGetTileLocalCtaIndex, DP2TileSK)
{
    /*
    The StreamKTilePartitionerBaseConfigDP2TileSK has the following form:
    - tiles in the C tensor: 7
    - iters_per_tile: 3
    - grid: 3
    - dp_tiles: 3
    - sk_tiles: 4
    - iters_per_sk_cta: 2
    - extra_iters: 2

    The tiles with iters are as follows:

    tile_idx: ____0___|___1___|___2___|___3___|___4___|____5____|____6____|
    tile_iter:| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
              |   |   |   |   |   |   |   |   |   |   |    |    |    |    |
              |<-------DP Tiles------>|<------------SK Tiles------------->|

    From the above configuration, we get the following:
    - SK CTA 0: tile_iter_start is 6 with local CTA index of 0 in tile 3
    - SK CTA 0: tile_iter_start is 8 with local CTA index of 0 in tile 4
    - SK CTA 1: tile_iter_start is 8 with local CTA index of 1 in tile 4
    - SK CTA 1: tile_iter_start is 10 with local CTA index of 0 in tile 5
    - SK CTA 2: tile_iter_start is 12 with local CTA index of 0 in tile 6
    */

    // Now we create a vector of triplets (tile_iter_start, cta_idx, tile_local_cta_idx) to test
    std::vector<std::array<ck_tile::index_t, 3>> sk_triplets{
        {6, 0, 0}, {8, 0, 0}, {8, 1, 1}, {10, 1, 0}, {12, 2, 0}};

    for(const auto& triplet : sk_triplets)
    {
        const auto& [tile_iter_start, cta_idx, tile_local_cta_idx] = triplet;
        test_get_tile_local_cta_idx<StreamKTilePartitionerBaseConfigDP2TileSK>(
            tile_iter_start, cta_idx, tile_local_cta_idx);
    }
}

TEST(StreamKTilePartitionerBaseGetKSize, NoRemainderTiles)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigRemainderAlongK;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel =
        KernelWrapperSpecialized<TilePartitioner, StreamKTilePartitionerBaseMethodId::GET_K_SIZE>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};
    ck_tile::DeviceMem k_size_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t num_macro_tiles = 2;
    ck_tile::index_t local_iter_end  = 2;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(num_macro_tiles,
                                        local_iter_end,
                                        Config::UNUSED,
                                        k_size_dev.GetDeviceBuffer(),
                                        nullptr,
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t k_size;
    k_size_dev.FromDevice(&k_size);

    /*
    In the StreamKTilePartitionerBaseConfigRemainderAlongK config, workgroup 0 is assigned the first
    2 macro tile along K. Both of these macro tiles are MPerBlock x KPerBlock. So, the k_size is
    K_TILE * 2. (See the struct definition for a detailed diagram.)
    */
    EXPECT_EQ(k_size, Config::K_TILE * 2);
}

TEST(StreamKTilePartitionerBaseGetKSize, RemainderTiles)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigRemainderAlongK;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel =
        KernelWrapperSpecialized<TilePartitioner, StreamKTilePartitionerBaseMethodId::GET_K_SIZE>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};
    ck_tile::DeviceMem k_size_dev(sizeof(ck_tile::index_t));
    ck_tile::index_t num_macro_tiles = 1;
    ck_tile::index_t local_iter_end  = 3;

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(num_macro_tiles,
                                        local_iter_end,
                                        Config::UNUSED,
                                        k_size_dev.GetDeviceBuffer(),
                                        nullptr,
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t k_size;
    k_size_dev.FromDevice(&k_size);

    /*
    In the StreamKTilePartitionerBaseConfigRemainderAlongK config, workgroup 1 is assigned the final
    macro tile along K. This macro tiles is MPerBlock x (K % K_TILE). So, the k_size is
    K % K_TILE. (See the struct definition for a detailed diagram.)
    */
    EXPECT_EQ(k_size, Config::K % Config::K_TILE);
}

// Persistent
TEST(StreamKTilePartitioner_PersistentConstructor, SKOnly)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::
        StreamKTilePartitioner<Config::GemmShape, ck_tile::StreamKReductionStrategy::Atomic, true>
            tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerV2PersistentExpected expected_values{0, 0, 3};
    validate_streamk_persistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_PersistentConstructor, DPOnly)
{
    using Config = StreamKTilePartitionerBaseConfigDPOnly;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerV2PersistentExpected expected_values{2, 0, 3};
    validate_streamk_persistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_PersistentConstructor, DP2TileSK)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerV2PersistentExpected expected_values{1, 0, 3};
    validate_streamk_persistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_PersistentConstructor, EdgeCase)
{
    using Config = StreamKTilePartitionerBaseConfigEdgeCase;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerV2PersistentExpected expected_values{0, 1, 4};
    validate_streamk_persistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_GridSize_Persistent, SKOnly)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    const auto g = tile_partitioner.grid_size();
    EXPECT_EQ(g.x, Config::MAX_ACTIVE_WGS);
}

TEST(StreamKTilePartitioner_GridSize_Persistent, EdgeCase)
{
    using Config = StreamKTilePartitionerBaseConfigEdgeCase;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    true>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    const auto g = tile_partitioner.grid_size();
    EXPECT_EQ(g.x, 1);
}

// Non-Persistent Tests
TEST(StreamKTilePartitioner_NonPersistentConstructor, SKOnly)
{
    using Config = StreamKTilePartitionerBaseConfigSKOnly;

    ck_tile::
        StreamKTilePartitioner<Config::GemmShape, ck_tile::StreamKReductionStrategy::Atomic, false>
            tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerV2NonPersistentExpected expected_values{0, 0, 0, 3};
    validate_streamk_nonpersistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_NonPersistentConstructor, DPOnly)
{
    using Config = StreamKTilePartitionerBaseConfigDPOnly;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerV2NonPersistentExpected expected_values{6, 0, 6, 3};
    validate_streamk_nonpersistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_NonPersistentConstructor, DP2TileSK)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerV2NonPersistentExpected expected_values{3, 0, 3, 3};
    validate_streamk_nonpersistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_NonPersistentConstructor, EdgeCase)
{
    using Config = StreamKTilePartitionerBaseConfigEdgeCase;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    StreamKTilePartitionerV2NonPersistentExpected expected_values{1, 0, 1, 4};
    validate_streamk_nonpersistent<Config::GemmShape>(expected_values, tile_partitioner);
}

TEST(StreamKTilePartitioner_GridSize_NonPersistent, DP2TileSK)
{
    using Config = StreamKTilePartitionerBaseConfigDP2TileSK;

    ck_tile::StreamKTilePartitioner<typename Config::GemmShape,
                                    ck_tile::StreamKReductionStrategy::Atomic,
                                    false>
        tile_partitioner{Config::M, Config::N, Config::K, Config::MAX_ACTIVE_WGS};

    const auto g = tile_partitioner.grid_size();
    EXPECT_EQ(g.x, 6);
}
