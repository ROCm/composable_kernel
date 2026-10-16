// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tail coverage for the CompAsyncV2 pipeline (two prefetch stages).
// num_loop 1 (tail One, no hot loop), 2 (tail Two, no hot loop), 3, 5 and 7 (tail One + hot
// loop), 4 and 32 (tail Two + hot loop). A tail that skips a K tile, an async K-tile read past
// the end of A/B or a missing async LDS fence shows up as wrong results or a memory fault.

TYPED_TEST(TEST_SUITE_NAME, UnpaddedTailCoverage)
{
    run_tail_coverage(
        TestFixture::K_Tile,
        {{512, 512}, {512, 1024}, {1024, 512}, {1024, 1024}},
        0,
        [this](int M, int N, int K) { this->template Run<false, false, false>(M, N, K); });
}

// Unpadded kernels also accept an M that is not a tile multiple when A and C are row major.
// The last M block then reads past the end of A, so on gfx125 some of its lanes skip their
// async loads and the hot-loop wait must drain fully for that block.
TYPED_TEST(TEST_SUITE_NAME, UnpaddedMTailCoverage)
{
    run_tail_coverage(
        TestFixture::K_Tile, {{300, 512}, {776, 256}}, 0, [this](int M, int N, int K) {
            this->template Run<false, false, false>(M, N, K);
        });
}

// Padded shapes: M is not a tile multiple and K ends half way into the last K tile, so
// num_loop = ceil(K / K_Tile) takes the same values as in the unpadded test. N is kept a
// tile multiple: a padded N combined with a padded K can fault in the async pipelines
// (CompAsync and CompAsyncV2 alike, independent of the tail handling tested here), which
// is tracked separately.
TYPED_TEST(TEST_SUITE_NAME, PaddedTailCoverage)
{
    constexpr int KT = TestFixture::K_Tile;
    run_tail_coverage(KT, {{300, 512}, {776, 256}}, KT / 2, [this](int M, int N, int K) {
        this->template Run<true, true, true>(M, N, K);
    });
}
