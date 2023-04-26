#pragma once

TEST_P(RCR_F16_F16_F16, TinyCases)
{
    const std::vector<int> Ms{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    const std::vector<int> Ns(Ms.size(), 768);
    const std::vector<int> Ks(Ms.size(), 4068);
    const std::vector<int> StrideAs(Ms.size(), 4608);
    const std::vector<int> StrideBs(Ms.size(), 4608);
    const std::vector<int> StrideCs(Ms.size(), 768);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_F16_F16_F16, TinyCases)
{
    const std::vector<int> Ms{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    const std::vector<int> Ns(Ms.size(), 4608);
    const std::vector<int> Ks(Ms.size(), 384);
    const std::vector<int> StrideAs(Ms.size(), 384);
    const std::vector<int> StrideBs(Ms.size(), 4608);
    const std::vector<int> StrideCs(Ms.size(), 4608);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_F16_F16_F16, SmallCases)
{
    const std::vector<int> Ms{2, 1, 1, 1, 1, 1, 3, 4, 3, 5, 2, 4, 2, 1, 0, 1};
    const std::vector<int> Ns(Ms.size(), 768);
    const std::vector<int> Ks(Ms.size(), 4068);
    const std::vector<int> StrideAs(Ms.size(), 4608);
    const std::vector<int> StrideBs(Ms.size(), 4608);
    const std::vector<int> StrideCs(Ms.size(), 768);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_F16_F16_F16, SmallCases)
{
    const std::vector<int> Ms{2, 1, 1, 1, 1, 1, 3, 4, 3, 5, 2, 4, 2, 1, 0, 1};
    const std::vector<int> Ns(Ms.size(), 4608);
    const std::vector<int> Ks(Ms.size(), 384);
    const std::vector<int> StrideAs(Ms.size(), 384);
    const std::vector<int> StrideBs(Ms.size(), 4608);
    const std::vector<int> StrideCs(Ms.size(), 4608);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_F16_F16_F16, MidCases)
{
    const std::vector<int> Ms{
        167, 183, 177, 181, 153, 139, 156, 173, 163, 150, 204, 184, 168, 156, 168, 148};
    const std::vector<int> Ns(Ms.size(), 768);
    const std::vector<int> Ks(Ms.size(), 4068);
    const std::vector<int> StrideAs(Ms.size(), 4608);
    const std::vector<int> StrideBs(Ms.size(), 4608);
    const std::vector<int> StrideCs(Ms.size(), 768);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_F16_F16_F16, MidCases)
{
    const std::vector<int> Ms{
        167, 183, 177, 181, 153, 139, 156, 173, 163, 150, 204, 184, 168, 156, 168, 148};
    const std::vector<int> Ns(Ms.size(), 4608);
    const std::vector<int> Ks(Ms.size(), 384);
    const std::vector<int> StrideAs(Ms.size(), 384);
    const std::vector<int> StrideBs(Ms.size(), 4608);
    const std::vector<int> StrideCs(Ms.size(), 4608);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}
