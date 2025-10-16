#pragma once

// Test with persistent mode enabled for preshuffle
TYPED_TEST(TestCkTileGroupedGemmPreshuffle, PreshufflePersistentBasic)
{
    const int group_count = 4;
    const int kbatch      = 1;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(128 + 64 * i);
        Ns.push_back(128 + 64 * i);
        Ks.push_back(256 + 128 * i);

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Cs.push_back(Ns[i]);
    }

    this->RunPersistent(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}

// Test with persistent mode and different problem sizes
TYPED_TEST(TestCkTileGroupedGemmPreshuffle, PreshufflePersistentVaried)
{
    const int group_count = 3;
    const int kbatch      = 1;
    std::vector<int> Ms = {64, 96, 128};
    std::vector<int> Ns = {64, 128, 192};
    std::vector<int> Ks = {128, 256, 384};
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;

    for(int i = 0; i < group_count; i++)
    {
        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Cs.push_back(Ns[i]);
    }

    this->RunPersistent(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}

// Test with persistent mode and large problem sizes
TYPED_TEST(TestCkTileGroupedGemmPreshuffle, PreshufflePersistentLarge)
{
    const int group_count = 2;
    const int kbatch      = 1;
    std::vector<int> Ms = {256, 512};
    std::vector<int> Ns = {256, 512};
    std::vector<int> Ks = {512, 1024};
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;

    for(int i = 0; i < group_count; i++)
    {
        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Cs.push_back(Ns[i]);
    }

    this->RunPersistent(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}
