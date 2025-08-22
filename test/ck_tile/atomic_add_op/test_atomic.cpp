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

template <typename DataType_>
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
            std::cerr << "n size should be multiple of dword_bytes" << std::endl;
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
        using Vector     = ck_tile::sequence<1, dword_bytes / sizeof(XDataType)>;

        std::cout << "Vector size / Yield per thread = " << (dword_bytes / sizeof(XDataType))
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

class TestAtomicKernelHalf : public TestAtomicKernel<ck_tile::half_t>
{
};

TEST_P(TestAtomicKernelHalf, TestCorrectness)
{
    auto [M, N] = GetParam();
    this->RunTest({M, N});
}

INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelHalf,
                         ::testing::Values(std::tuple{64, 8}));
