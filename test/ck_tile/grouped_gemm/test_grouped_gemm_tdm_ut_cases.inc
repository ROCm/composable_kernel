// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Groups of increasing size. M and N are multiples of the 64x64 block tile and K is a multiple
// of 128, which is a multiple of the K block tile for every dtype in the suite (32 for 16-bit,
// 128 for 8-bit), so no group has a K tail or K < K_Tile.
TYPED_TEST(TEST_CKTILE_GGEMM_TDM_SUITE_NAME, Basic)
{
    const int group_count = 8;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(64 + 64 * i);
        Ns.push_back(128 + 64 * i);
        Ks.push_back(128 * (1 + i % 4));
    }

    this->RunOnDevice(Ms, Ns, Ks, group_count);
}

// M and N that are not multiples of the 64x64 block tile: the kernel is built without M/N
// padding, so the partial edge tiles rely on TDM clipping the A/B loads and the C store. M and N
// stay multiples of 8 elements (row pitch 16 bytes for 16-bit types, 8 bytes for 8-bit types);
// K is a multiple of 128.
TYPED_TEST(TEST_CKTILE_GGEMM_TDM_SUITE_NAME, RaggedMN)
{
    const int group_count     = 4;
    const std::vector<int> Ms = {8, 200, 72, 136};
    const std::vector<int> Ns = {264, 40, 8, 1000};
    const std::vector<int> Ks = {128, 256, 384, 128};

    this->RunOnDevice(Ms, Ns, Ks, group_count);
}

// Long K loops exercise the TDM ping-pong LDS schedule. Every K is a multiple of the K block
// tile; K = 1024 + K_Tile and 4096 + K_Tile give an odd number of K iterations for every dtype
// (16-bit: 33 and 129, 8-bit: 9 and 33). The K schedule does not depend on M and N, so each group
// is one or two 64x64 tiles to keep the CPU reference cheap.
TYPED_TEST(TEST_CKTILE_GGEMM_TDM_SUITE_NAME, LargeK)
{
    constexpr int KTile       = TestFixture::ActiveKernelParam::K_Tile;
    const int group_count     = 4;
    const std::vector<int> Ms = {128, 64, 64, 64};
    const std::vector<int> Ns = {64, 64, 64, 64};
    const std::vector<int> Ks = {1024, 1024 + KTile, 4096, 4096 + KTile};

    this->RunOnDevice(Ms, Ns, Ks, group_count);
}

// Groups with M=0 or K=0 are dropped by MakeKargs; the remaining groups must be unaffected and
// the K=0 groups must keep their zero-initialized output. Non-zero K values are multiples of 128.
TYPED_TEST(TEST_CKTILE_GGEMM_TDM_SUITE_NAME, EmptyGroup)
{
    const int group_count     = 8;
    const std::vector<int> Ms = {256, 0, 256, 256, 128, 0, 256, 192};
    const std::vector<int> Ns = {256, 256, 128, 256, 256, 256, 192, 256};
    const std::vector<int> Ks = {512, 512, 512, 0, 256, 512, 0, 384};

    this->RunOnDevice(Ms, Ns, Ks, group_count);
}

// K tails and K < K_Tile. K = 64 and 96 are below the 8-bit K_Tile (128), so they are a single
// partial tile on 8-bit; K = 1056 has a tail on 8-bit only; K = 432 has a tail on every dtype
// (16-bit K_Tile is 32).
TYPED_TEST(TEST_CKTILE_GGEMM_TDM_SUITE_NAME, KTail)
{
    const int group_count     = 4;
    const std::vector<int> Ms = {256, 128, 256, 192};
    const std::vector<int> Ns = {256, 256, 128, 256};
    const std::vector<int> Ks = {64, 96, 1056, 432};

    this->RunOnDevice(Ms, Ns, Ks, group_count);
}

// MakeKargs drops zero-sized groups but retains negative K. Reject a negative K in any group
// before querying the device, without allocating tensors or launching invalid TDM loads.
TYPED_TEST(TEST_CKTILE_GGEMM_TDM_SUITE_NAME, NegativeKRejected)
{
    constexpr ck_tile::index_t M = 256;
    constexpr ck_tile::index_t N = 256;
    constexpr ck_tile::index_t K = 512;
    constexpr ck_tile::index_t stride_A =
        std::is_same_v<typename TestFixture::ALayout, ck_tile::tensor_layout::gemm::RowMajor> ? K
                                                                                              : M;
    constexpr ck_tile::index_t stride_B =
        std::is_same_v<typename TestFixture::BLayout, ck_tile::tensor_layout::gemm::RowMajor> ? N
                                                                                              : K;

    const typename TestFixture::grouped_gemm_kargs valid_desc{
        nullptr, nullptr, {}, nullptr, 1, M, N, K, stride_A, stride_B, {}, N};
    for(const ck_tile::index_t invalid_k : {-1, -TestFixture::ActiveKernelParam::K_Tile})
    {
        for(std::size_t group = 0; group < 3; ++group)
        {
            SCOPED_TRACE(::testing::Message() << "group=" << group << ", K=" << invalid_k);
            std::vector<typename TestFixture::grouped_gemm_kargs> gemm_descs(3, valid_desc);
            gemm_descs[group].K = invalid_k;
            EXPECT_FALSE(this->IsTdmArgumentSupported(gemm_descs));
        }
    }
}

// TdmEpilogue has no atomic path, so split-K must be rejected on the host. The same
// descriptors with k_batch = 1 must be accepted on a gfx125 device, so the rejection is
// attributable to k_batch. IsSupportedArgument checks k_batch before querying the device.
TYPED_TEST(TEST_CKTILE_GGEMM_TDM_SUITE_NAME, SplitKRejected)
{
    const int group_count = 4;

    const auto make_descs = [&](const ck_tile::index_t kbatch) {
        std::vector<typename TestFixture::grouped_gemm_kargs> gemm_descs;
        for(int i = 0; i < group_count; i++)
        {
            const ck_tile::index_t M        = 256;
            const ck_tile::index_t N        = 256;
            const ck_tile::index_t K        = 512;
            const ck_tile::index_t stride_A = std::is_same_v<typename TestFixture::ALayout,
                                                             ck_tile::tensor_layout::gemm::RowMajor>
                                                  ? K
                                                  : M;
            const ck_tile::index_t stride_B = std::is_same_v<typename TestFixture::BLayout,
                                                             ck_tile::tensor_layout::gemm::RowMajor>
                                                  ? N
                                                  : K;
            const ck_tile::index_t stride_C = N;
            gemm_descs.push_back({nullptr,
                                  nullptr,
                                  {/*ds_ptr*/},
                                  nullptr,
                                  kbatch,
                                  M,
                                  N,
                                  K,
                                  stride_A,
                                  stride_B,
                                  {/*stride_Ds*/},
                                  stride_C});
        }
        return gemm_descs;
    };

    EXPECT_FALSE(this->IsTdmArgumentSupported(make_descs(2)));

    if(!ck_tile::is_gfx125_supported())
    {
        GTEST_SKIP() << "k_batch = 1 acceptance check requires a gfx125 device.";
    }
    EXPECT_TRUE(this->IsTdmArgumentSupported(make_descs(1)));
}
