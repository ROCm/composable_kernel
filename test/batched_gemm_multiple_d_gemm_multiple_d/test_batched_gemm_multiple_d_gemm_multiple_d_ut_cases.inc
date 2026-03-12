
TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test) { this->Run(); }

TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test_PadM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {136, 128, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test_PadN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 136, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test_PadK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 40, 128, 1},
        {128, 128, 136, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test_PadO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 136, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test_OddM)
{
    if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
    {
        GTEST_SKIP() << "Odd-sizes only supported on WMMA instances.";
    }

    this->lengths_ = std::vector<std::vector<int>>{
        {129, 128, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test_OddN)
{
    if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
    {
        GTEST_SKIP() << "Odd-sizes only supported on WMMA instances.";
    }

    this->lengths_ = std::vector<std::vector<int>>{
        {128, 129, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test_OddK)
{
    if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
    {
        GTEST_SKIP() << "Odd-sizes only supported on WMMA instances.";
    }

    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 33, 128, 1},
        {128, 128, 129, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, Test_OddO)
{
    if(!(ck::is_gfx11_supported() || ck::is_gfx12_supported()))
    {
        GTEST_SKIP() << "Odd-sizes only supported on WMMA instances.";
    }

    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 129, 1},
    };
    this->Run();
}
