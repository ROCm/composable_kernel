#pragma once

TYPED_TEST(TestCkTileGroupedGemmQuant, Basic)
{
    const int group_count = 8;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;
    std::vector<int> stride_AQs;
    std::vector<int> stride_BQs;
    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256 + 256 * i);
        Ns.push_back(256 + 512 * i);
        Ks.push_back(512 + 128 * i);

        stride_As.push_back(0);
        stride_Bs.push_back(0);
        stride_Cs.push_back(0);
        stride_AQs.push_back(0);
        stride_BQs.push_back(0);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, stride_AQs, stride_BQs, group_count);
}
