// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Batched Stream-K: the output tiles of all batches form one tile space, so data-parallel tiles
// and Stream-K segments run across batch boundaries. A fixed workgroup budget (num_cu with
// occupancy one) keeps the partition independent of the device.

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_OneTilePerBatch)
{
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // 11 one-tile batches on 8 workgroups: Stream-K only, and every segment boundary inside a
    // tile is a partial tile of some batch.
    this->Run(M_Tile, N_Tile, 10 * K_Tile, 0, 0, 0, /*num_cu=*/8, /*require_split=*/true, 11);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_DataParallelAndStreamK)
{
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // 7 batches of 2 x 3 tiles on 8 workgroups: 32 data-parallel tiles, then 10 Stream-K tiles
    // that span the last two batches.
    this->Run(2 * M_Tile, 3 * N_Tile, 6 * K_Tile, 0, 0, 0, /*num_cu=*/8, /*require_split=*/true, 7);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_PaddedTiles)
{
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // Partial M, N and K tiles in every batch: 5 batches of 3 x 2 tiles on 7 workgroups, then the
    // same on the device's workgroup budget.
    this->Run(2 * M_Tile + 16, N_Tile + 32, 7 * K_Tile + 8, 0, 0, 0, /*num_cu=*/7, true, 5);
    this->Run(2 * M_Tile + 16, N_Tile + 32, 7 * K_Tile + 8, 0, 0, 0, 0, false, 5);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_SingleBatch)
{
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // One batch is the single-GEMM partition.
    this->Run(M_Tile, 3 * N_Tile, 9 * K_Tile, 0, 0, 0, /*num_cu=*/4, /*require_split=*/true, 1);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_RejectsOverlappingPaddedLeadingStride)
{
    using ALayout                     = typename TestFixture::ALayout;
    using BLayout                     = typename TestFixture::BLayout;
    using CLayout                     = typename TestFixture::CLayout;
    using RowMajor                    = ck_tile::tensor_layout::gemm::RowMajor;
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    const ck_tile::index_t M = M_Tile;
    const ck_tile::index_t N = N_Tile;
    const ck_tile::index_t K = 2 * K_Tile;

    constexpr bool a_row = std::is_same_v<ALayout, RowMajor>;
    constexpr bool b_row = std::is_same_v<BLayout, RowMajor>;
    constexpr bool c_row = std::is_same_v<CLayout, RowMajor>;

    const ck_tile::index_t inner_a = a_row ? K : M;
    const ck_tile::index_t outer_a = a_row ? M : K;
    const ck_tile::index_t inner_b = b_row ? N : K;
    const ck_tile::index_t outer_b = b_row ? K : N;
    const ck_tile::index_t inner_c = c_row ? N : M;
    const ck_tile::index_t outer_c = c_row ? M : N;

    // Padded leading strides (2x contiguous dim): for M=N=128 row-major C, stride_C=256 so the
    // dense product M*N=16384 overlaps 8,192 elements between adjacent batches; the minimum
    // non-overlapping batch stride is (outer - 1) * stride + inner = 32640.
    const ck_tile::index_t pad_sa = 2 * inner_a;
    const ck_tile::index_t pad_sb = 2 * inner_b;
    const ck_tile::index_t pad_sc = 2 * inner_c;

    const ck_tile::long_index_t min_bsa =
        static_cast<ck_tile::long_index_t>(outer_a - 1) * pad_sa + inner_a;
    const ck_tile::long_index_t min_bsb =
        static_cast<ck_tile::long_index_t>(outer_b - 1) * pad_sb + inner_b;
    const ck_tile::long_index_t min_bsc =
        static_cast<ck_tile::long_index_t>(outer_c - 1) * pad_sc + inner_c;

    const auto* a = reinterpret_cast<const void*>(std::uintptr_t{0x1000});
    const auto* b = reinterpret_cast<const void*>(std::uintptr_t{0x1000});
    auto* c       = reinterpret_cast<void*>(std::uintptr_t{0x1000});

    const ck_tile::StreamKHostArgs valid_min{
        a, b, c, M, N, K, pad_sa, pad_sb, pad_sc, min_bsa, min_bsb, min_bsc, /*batch_count=*/2};
    EXPECT_TRUE(this->IsArgSupported(valid_min, /*num_cu=*/8));

    const ck_tile::StreamKHostArgs valid_slab{a,
                                              b,
                                              c,
                                              M,
                                              N,
                                              K,
                                              pad_sa,
                                              pad_sb,
                                              pad_sc,
                                              static_cast<ck_tile::long_index_t>(outer_a) * pad_sa,
                                              static_cast<ck_tile::long_index_t>(outer_b) * pad_sb,
                                              static_cast<ck_tile::long_index_t>(outer_c) * pad_sc,
                                              /*batch_count=*/2};
    EXPECT_TRUE(this->IsArgSupported(valid_slab, /*num_cu=*/8));

    // Dense shape products (M*K, K*N, M*N) and min_bs - 1 overlap when leading strides are padded.
    auto bad_c           = valid_min;
    bad_c.batch_stride_E = static_cast<ck_tile::long_index_t>(M) * N;
    EXPECT_FALSE(this->IsArgSupported(bad_c, /*num_cu=*/8));
    bad_c.batch_stride_E = min_bsc - 1;
    EXPECT_FALSE(this->IsArgSupported(bad_c, /*num_cu=*/8));

    auto bad_a           = valid_min;
    bad_a.batch_stride_A = static_cast<ck_tile::long_index_t>(M) * K;
    EXPECT_FALSE(this->IsArgSupported(bad_a, /*num_cu=*/8));
    bad_a.batch_stride_A = min_bsa - 1;
    EXPECT_FALSE(this->IsArgSupported(bad_a, /*num_cu=*/8));

    auto bad_b           = valid_min;
    bad_b.batch_stride_B = static_cast<ck_tile::long_index_t>(K) * N;
    EXPECT_FALSE(this->IsArgSupported(bad_b, /*num_cu=*/8));
    bad_b.batch_stride_B = min_bsb - 1;
    EXPECT_FALSE(this->IsArgSupported(bad_b, /*num_cu=*/8));
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_Batched_RejectsPreValidationTileAndIterOverflow)
{
    using Kernel                      = typename TestFixture::template KernelType<>;
    using ALayout                     = typename TestFixture::ALayout;
    using BLayout                     = typename TestFixture::BLayout;
    using CLayout                     = typename TestFixture::CLayout;
    using RowMajor                    = ck_tile::tensor_layout::gemm::RowMajor;
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    const auto* a = reinterpret_cast<const void*>(std::uintptr_t{0x1000});
    const auto* b = reinterpret_cast<const void*>(std::uintptr_t{0x1000});
    auto* c       = reinterpret_cast<void*>(std::uintptr_t{0x1000});

    auto make_packed_args =
        [&](ck_tile::index_t m, ck_tile::index_t n, ck_tile::index_t k, ck_tile::index_t batches) {
            const ck_tile::index_t sa = std::is_same_v<ALayout, RowMajor> ? k : m;
            const ck_tile::index_t sb = std::is_same_v<BLayout, RowMajor> ? n : k;
            const ck_tile::index_t sc = std::is_same_v<CLayout, RowMajor> ? n : m;
            return ck_tile::StreamKHostArgs{a,
                                            b,
                                            c,
                                            m,
                                            n,
                                            k,
                                            sa,
                                            sb,
                                            sc,
                                            static_cast<ck_tile::long_index_t>(m) * k,
                                            static_cast<ck_tile::long_index_t>(k) * n,
                                            static_cast<ck_tile::long_index_t>(m) * n,
                                            batches};
        };

    // 256 x 256 = 65536 tiles per batch x 65536 batches = 2^32 > INT32_MAX: MakeKernelArgs runs
    // the TilePartitioner constructor before IsSupportedArgument and must not overflow index_t.
    const ck_tile::index_t M = 256 * M_Tile;
    const ck_tile::index_t N = 256 * N_Tile;
    const ck_tile::index_t K = 2 * K_Tile;

    const auto tile_overflow = make_packed_args(M, N, K, /*batches=*/65536);
    const auto kargs_tiles   = Kernel::MakeKernelArgs(tile_overflow, /*num_cu=*/8, /*occupancy=*/1);
    EXPECT_FALSE(kargs_tiles.tile_partitioner.is_valid());
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs_tiles));

    // num_tiles = 65536 fits in index_t, but num_tiles * iters_per_tile = 2^32 exceeds index_t.
    const auto iter_overflow = make_packed_args(M, N, 65536 * K_Tile, /*batches=*/1);
    const auto kargs_iters   = Kernel::MakeKernelArgs(iter_overflow, /*num_cu=*/8, /*occupancy=*/1);
    EXPECT_FALSE(kargs_iters.tile_partitioner.is_valid());
    EXPECT_FALSE(Kernel::IsSupportedArgument(kargs_iters));
}
