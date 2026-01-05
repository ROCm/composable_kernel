// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once


TYPED_TEST(TestGroupedGemm, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 512;
    constexpr int K = 1024;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemm, TestLargeKBatch)
{
    // In some cases Split K is not supported. Running this test would fail since no instance will
    // be supported, so we skip the test
    if(!this->IsSplitKSupported())
        GTEST_SKIP() << "Split-K not supported for for the current configuration (FP16/BF16 on "
                        "GFX11, or using CDE element-wise operation)";

    const std::vector<int> Ms{188, 210};
    constexpr int N = 4096;
    constexpr int K = 4096;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->k_batches_ = {32, 64};

    this->Run(Ms, Ns, Ks);
}
