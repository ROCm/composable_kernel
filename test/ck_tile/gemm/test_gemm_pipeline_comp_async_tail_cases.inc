// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unpadded shapes: K is a tile multiple chosen to hit each tail path of the async pipeline.
// num_loop 1 (tail One), 2 (tail Two, no hot loop), 3 (tail Three, no hot loop),
// 4 and 32 (tail Two + hot loop), 5 and 7 (tail Three + hot loop; 7 iterates the hot loop
// more than once). Any async K-tile over-read past the end of A/B or a missing async LDS
// fence in a tail shows up as wrong results or an illegal memory access.
// Keep k_batch = 1 so the loop counts are exact on both XDL and WMMA.
TYPED_TEST(TEST_SUITE_NAME, UnpaddedTailCoverage)
{
    run_tail_coverage(TestFixture::K_Tile,
                      {{512, 512}, {512, 1024}, {1024, 512}, {1024, 1024}},
                      0,
                      [this](int M, int N, int K) {
                          this->template RunSingle<false, false, false, false>(M, N, K, 0, 0, 0, 1);
                      });
}
