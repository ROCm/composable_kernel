
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
    this->lengths_ = std::vector<std::vector<int>>{
        {129, 128, 32, 128, 1},
    };
    this->Run();
}

// NOTE: Disabled as not supported for B1 ColMajor
TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, DISABLED_Test_OddN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 129, 32, 128, 1},
    };
    this->Run();
}

// NOTE: Disabled as no instances added for small transfer vector sizes
TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, DISABLED_Test_OddK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 33, 128, 1},
        {128, 128, 129, 128, 1},
    };
    this->Run();
}

// If kernel B1Layout is RowMajor, expect not to support odd O size
TYPED_TEST(TestBatchedGemmMultipleDGemmMultipleD, DISABLED_Test_OddO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 129, 1},
    };
    this->Run();
}
