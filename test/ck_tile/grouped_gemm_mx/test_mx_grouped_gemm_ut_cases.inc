// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TestCkTileMxGroupedGemm, Basic)
{
    const int group_count = 4;
    const int kbatch      = 1;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256 + 256 * i);
        Ns.push_back(256 + 512 * i);
        Ks.push_back(512 + 128 * i);
    }

    this->Run(Ms, Ns, Ks, kbatch, group_count);
}

