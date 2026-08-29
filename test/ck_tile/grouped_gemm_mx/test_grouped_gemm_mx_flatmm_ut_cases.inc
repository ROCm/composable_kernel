// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// =================================================================================
// Test cases for GroupedMXFlatmmKernel: mixed-K coverage.
//
// The MX FLATMM pipelines (both MXFlatmmPipelineAGmemBGmemCRegV1 and
// WeightPreshufflePipelineAGmemBGmemCRegTDM) dispatch (HasHotLoop, TailNum)
// at runtime inside the pipeline body based on the per-tile num_loop. So
// groups within a single Run() call may use any mix of K values -- there is
// no same-class constraint at the test level.
//
//   PrefetchStages = 2
//   BlockHasHotloop(num_loop)   = (num_loop > 2)
//   GetBlockLoopTailNum(n_loop) = (n_loop % 2 == 0) ? Even : Odd
//   K_Tile = 256, so num_loop = ceil(K / K_Tile)
//
//   | Class | HasHotLoop | TailNum | num_loop      | Valid K (K_Tile=256)      |
//   |-------|------------|---------|---------------|---------------------------|
//   |  A    | false      | Odd     | {1}           | {256}                     |
//   |  B    | false      | Even    | {2}           | {512}                     |
//   |  C    | true       | Odd     | {3, 5, 7, ...}| {768, 1280, 1792, ...}    |
//   |  D    | true       | Even    | {4, 6, 8, ...}| {1024, 1536, 2048, ...}   |
//
// Dimension constraints:
//   M must be a multiple of M_Tile = 128 (all configs).
//   N must be a multiple of 512 to satisfy MXfp4_FlatmmConfig16's N_Tile=512;
//     this is a superset of MXFlatmmConfigBase16 / MXFlatmmConfigBase32TDM
//     (N_Tile=256), so the same test cases cover both non-TDM and TDM binaries.
//   K must be a multiple of K_Tile = 256 and >= 256 (smallest non-trivial size).
// =================================================================================

// init_method: 1 = constant init (exact-integer oracle, near-exact tolerance),
//              0 = random init (uniform values, 1e-2 tolerance). Each case is
// run in both regimes; the constant variant pins exact K accumulation while the
// random variant exercises realistic value distributions.
constexpr int init_const  = 1;
constexpr int init_random = 0;

TYPED_TEST(TestGroupedGemmMXFlatmm, Basic_4Groups)
{
    // K values hit one class each: 256 (A), 512 (B), 768 (C), 1024 (D).
    this->Run({128, 256, 384, 512}, {512, 1024, 512, 1024}, {256, 512, 768, 1024}, 4, init_const);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Basic_4Groups_Random)
{
    this->Run({128, 256, 384, 512}, {512, 1024, 512, 1024}, {256, 512, 768, 1024}, 4, init_random);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Single_Group) { this->Run({128}, {512}, {512}, 1, init_const); }

TYPED_TEST(TestGroupedGemmMXFlatmm, Single_Group_Random)
{
    this->Run({128}, {512}, {512}, 1, init_random);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Two_Groups_Same_Size)
{
    this->Run({128, 128}, {512, 512}, {256, 256}, 2, init_const);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Two_Groups_Same_Size_Random)
{
    this->Run({128, 128}, {512, 512}, {256, 256}, 2, init_random);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Eight_Groups)
{
    // K cycling through {256, 512, 768, 1024, 1280, 1536, 1792, 2048} -- every
    // class hit, deeply mixed. Exercises per-tile runtime dispatch across all 4
    // (HasHotLoop, TailNum) variants in one launch, validating grouped-GEMM-MX
    // correctness when groups of different K classes coexist.
    this->Run({128, 256, 384, 512, 128, 512, 256, 384},
              {512, 1024, 512, 1024, 512, 1024, 1024, 512},
              {256, 512, 768, 1024, 1280, 2048, 1536, 1792},
              8,
              init_const);
}

TYPED_TEST(TestGroupedGemmMXFlatmm, Eight_Groups_Random)
{
    this->Run({128, 256, 384, 512, 128, 512, 256, 384},
              {512, 1024, 512, 1024, 512, 1024, 1024, 512},
              {256, 512, 768, 1024, 1280, 2048, 1536, 1792},
              8,
              init_random);
}
