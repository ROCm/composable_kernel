// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TEST_CKTILE_GGEMM_SUITE_NAME, Basic)
{
    const int group_count = 8;
    const int kbatch      = 1;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(64 + 64 * i);
        Ns.push_back(128 + 64 * i);
        Ks.push_back(64 + 32 * i);

        stride_As.push_back(0);
        stride_Bs.push_back(0);
        stride_Cs.push_back(0);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}

TYPED_TEST(TEST_CKTILE_GGEMM_SUITE_NAME, SplitK)
{
    const int group_count = 8;
    const int kbatch      = 2;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(64 + 64 * i);
        Ns.push_back(128 + 64 * i);
        Ks.push_back(64 + 32 * i);

        stride_As.push_back(0);
        stride_Bs.push_back(0);
        stride_Cs.push_back(0);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}

// Verify that groups with M=0 are silently skipped (primary MoE scenario where some
// experts receive zero tokens) and that non-zero groups produce correct results.
TYPED_TEST(TEST_CKTILE_GGEMM_SUITE_NAME, ZeroM)
{
    const int group_count = 8;
    const int kbatch      = 1;

    const std::vector<int> Ms = {256, 0, 256, 0, 256, 256, 0, 256};
    const std::vector<int> Ns = {256, 256, 256, 256, 256, 256, 256, 256};
    const std::vector<int> Ks = {512, 512, 512, 512, 512, 512, 512, 512};
    std::vector<int> stride_As(group_count, 0);
    std::vector<int> stride_Bs(group_count, 0);
    std::vector<int> stride_Cs(group_count, 0);

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}

// Verify that groups with K=0 produce all-zero output and that surrounding groups
// with non-zero K are unaffected.
TYPED_TEST(TEST_CKTILE_GGEMM_SUITE_NAME, ZeroK)
{
    const int group_count = 8;
    const int kbatch      = 1;

    const std::vector<int> Ms = {256, 256, 256, 256, 256, 256, 256, 256};
    const std::vector<int> Ns = {256, 256, 256, 256, 256, 256, 256, 256};
    const std::vector<int> Ks = {512, 512, 512, 0, 0, 512, 512, 512};
    std::vector<int> stride_As(group_count, 0);
    std::vector<int> stride_Bs(group_count, 0);
    std::vector<int> stride_Cs(group_count, 0);

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}

// Verify that a mix of M=0, N=0, and K=0 groups all behave correctly together.
TYPED_TEST(TEST_CKTILE_GGEMM_SUITE_NAME, ZeroMixed)
{
    const int group_count = 8;
    const int kbatch      = 1;

    const std::vector<int> Ms = {256, 0, 256, 256, 512, 256, 0, 256};
    const std::vector<int> Ns = {256, 256, 512, 0, 256, 256, 256, 256};
    const std::vector<int> Ks = {512, 512, 512, 512, 512, 0, 512, 512};
    std::vector<int> stride_As(group_count, 0);
    std::vector<int> stride_Bs(group_count, 0);
    std::vector<int> stride_Cs(group_count, 0);

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}
