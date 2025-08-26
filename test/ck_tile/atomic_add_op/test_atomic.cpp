// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "test_atomic.hpp"

struct AtomicKernelParam
{
    AtomicKernelParam(ck_tile::index_t m_, ck_tile::index_t n_) : m(m_), n(n_) {}
    ck_tile::index_t m;
    ck_tile::index_t n;
};

template <typename DataType_, ck_tile::index_t multiple_>
class TestAtomicKernel : public ::testing::TestWithParam<std::tuple<int, int>>
{
    protected:
    void RunTest(const AtomicKernelParam& params)
    {
        using XDataType = DataType_;

        ck_tile::index_t m = params.m;
        ck_tile::index_t n = params.n;

        std::cout << "Input Tensor Dimensions: " << m << ", " << n << std::endl;

        constexpr auto dword_bytes = 4;

        if(n % (dword_bytes / sizeof(XDataType)) != 0)
        {
            ASSERT_EQ(n % (multiple_ * (dword_bytes / sizeof(XDataType))), 0) << " Row dimension must be divisible by vector width";
        }

        // host tensor
        ck_tile::HostTensor<XDataType> x_host_ref({m, n});
        ck_tile::HostTensor<XDataType> x_host_dev({m, n});

        // device buffers
        ck_tile::DeviceMem x_dev_input(x_host_dev.get_element_space_size_in_bytes());
        x_dev_input.SetZero();
        x_host_ref.SetZero();

        using BlockWaves = ck_tile::sequence<2, 1>;
        using BlockTile  = ck_tile::sequence<64, 8>;
        using WaveTile   = ck_tile::sequence<64, 8>;
        using Vector     = ck_tile::sequence<1, multiple_ * (dword_bytes / sizeof(XDataType))>;

        std::cout << "Vector size / Yield per thread = " << multiple_ * (dword_bytes / sizeof(XDataType))
                  << std::endl;

        ck_tile::index_t kGridSize =
            ck_tile::integer_divide_ceil(m, BlockTile::at(ck_tile::number<0>{}));

        using Shape   = ck_tile::AtomicKernelShape<BlockWaves, BlockTile, WaveTile, Vector>;
        using Problem = ck_tile::AtomicKernelProblem<XDataType, Shape>;
        using Kernel  = ck_tile::AtomicKernel<Problem>;

        constexpr ck_tile::index_t kBlockSize  = 128;
        constexpr ck_tile::index_t kBlockPerCu = 1;

        launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                      ck_tile::make_kernel<kBlockPerCu>(
                          Kernel{},
                          kGridSize,
                          kBlockSize,
                          0,
                          static_cast<XDataType*>(x_dev_input.GetDeviceBuffer()),
                          m,
                          n));

        // host reference computation
        x_dev_input.FromDevice(x_host_dev.mData.data());
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                x_host_ref(i, j) = static_cast<XDataType>(1);
            }
        }
        bool pass = ck_tile::check_err(x_host_dev, x_host_ref);

        EXPECT_TRUE(pass);
    }
};

class TestAtomicKernelHalf_1 : public TestAtomicKernel<ck_tile::half_t, 1>
{
};

class TestAtomicKernelHalf_2 : public TestAtomicKernel<ck_tile::half_t, 2>
{
};

class TestAtomicKernelHalf_4 : public TestAtomicKernel<ck_tile::half_t, 4>
{
};

class TestAtomicKernelBF16_1 : public TestAtomicKernel<ck_tile::bf16_t, 1>
{
};

class TestAtomicKernelBF16_2 : public TestAtomicKernel<ck_tile::bf16_t, 2>
{
};

class TestAtomicKernelBF16_4 : public TestAtomicKernel<ck_tile::bf16_t, 4>
{
};

class TestAtomicKernelBF8_1 : public TestAtomicKernel<ck_tile::bf8_t, 1>
{
};

class TestAtomicKernelBF8_2 : public TestAtomicKernel<ck_tile::bf8_t, 2>
{
};

class TestAtomicKernelFP8_1 : public TestAtomicKernel<ck_tile::fp8_t, 1>
{
};

class TestAtomicKernelFP8_2 : public TestAtomicKernel<ck_tile::fp8_t, 2>
{
};

class TestAtomicKernelFloat_1 : public TestAtomicKernel<float, 1>
{
};

class TestAtomicKernelFloat_2 : public TestAtomicKernel<float, 2>
{
};

class TestAtomicKernelFloat_4 : public TestAtomicKernel<float, 4>
{
};

TEST_P(TestAtomicKernelHalf_1, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelHalf_2, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelHalf_4, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelBF16_1, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelBF16_2, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelBF16_4, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelBF8_1, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelBF8_2, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelFP8_1, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelFP8_2, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelFloat_1, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelFloat_2, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

TEST_P(TestAtomicKernelFloat_4, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelHalf_1,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelHalf_2,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));
                                           
INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelHalf_4,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));                                           

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelBF16_1,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelBF16_2,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));
                                           
INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelBF16_4,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelBF8_1,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelBF8_2,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelFP8_1,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelFP8_2,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));                                            

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelFloat_1,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));  

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelFloat_2,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));
                                           
INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelFloat_4,
                         ::testing::Values(std::tuple{64, 8}, 
                                           std::tuple{64, 16}, 
                                           std::tuple{64, 32}));                                           
