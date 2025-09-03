// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights
// reserved.

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

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
    struct AtomicKernelWaveSize64
    {
        using BlockWaves                             = ck_tile::sequence<2, 1>;
        using BlockTile                              = ck_tile::sequence<128, 8>;
        using WaveTile                               = ck_tile::sequence<64, 8>;
        static constexpr ck_tile::index_t kBlockSize = 128; // 2 waves * 64 lanes
    };

    struct AtomicKernelWaveSize32
    {
        using BlockWaves                             = ck_tile::sequence<2, 1>;
        using BlockTile                              = ck_tile::sequence<64, 8>;
        using WaveTile                               = ck_tile::sequence<32, 8>; // 32*2 == 64
        static constexpr ck_tile::index_t kBlockSize = 64; // 2 waves * 32 lanes
    };

    template <typename Config>
    void RunTestImpl_(const AtomicKernelParam& params, int require_warp_size, const char* tag)
    {
        // Device capability check & skip if wavesize mismatches
        int dev = 0;
        hipDeviceProp_t prop{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess)
        {
            GTEST_SKIP() << "[" << tag << "] hipGetDeviceProperties failed; skipping.";
        }
        if(prop.warpSize != require_warp_size)
        {
            GTEST_SKIP() << "[" << tag << "] Device warpSize=" << prop.warpSize << " (requires "
                         << require_warp_size << "); skipping.";
        }

        using XDataType = DataType_;

        const ck_tile::index_t m = params.m;
        const ck_tile::index_t n = params.n;

        std::cout << "[" << tag << "] Input Tensor Dimensions: " << m << ", " << n << std::endl;

        constexpr int dword_bytes = 4;
        const int base_vec        = dword_bytes / static_cast<int>(sizeof(XDataType));
        const int vec             = multiple_ * base_vec;

        ASSERT_EQ(n % vec, 0) << " Row dimension must be divisible by vector width: n=" << n
                              << " vec=" << vec << " (multiple=" << multiple_
                              << ", base_vec=" << base_vec << ")";

        // host tensors
        ck_tile::HostTensor<XDataType> x_host_ref({m, n});
        ck_tile::HostTensor<XDataType> x_host_dev({m, n});

        // device buffers
        ck_tile::DeviceMem x_dev_input(x_host_dev.get_element_space_size_in_bytes());
        x_dev_input.SetZero();
        x_host_ref.SetZero();

        using BlockWaves = typename Config::BlockWaves;
        using BlockTile  = typename Config::BlockTile;
        using WaveTile   = typename Config::WaveTile;
        using Vector     = ck_tile::sequence<1, vec>;

        // Compile-time sanity: BlockTile == WaveTile * BlockWaves
        static_assert(BlockTile::at(ck_tile::number<0>{}) ==
                          WaveTile::at(ck_tile::number<0>{}) * BlockWaves::at(ck_tile::number<0>{}),
                      "BlockTile.M must equal WaveTile.M * BlockWaves.M");
        static_assert(BlockTile::at(ck_tile::number<1>{}) ==
                          WaveTile::at(ck_tile::number<1>{}) * BlockWaves::at(ck_tile::number<1>{}),
                      "BlockTile.N must equal WaveTile.N * BlockWaves.N");

        std::cout << "[" << tag << "] Vector per thread = " << vec
                  << "  BlockWaves=" << BlockWaves::at(ck_tile::number<0>{}) << "x"
                  << BlockWaves::at(ck_tile::number<1>{})
                  << "  WaveTile=" << WaveTile::at(ck_tile::number<0>{}) << "x"
                  << WaveTile::at(ck_tile::number<1>{})
                  << "  BlockTile=" << BlockTile::at(ck_tile::number<0>{}) << "x"
                  << BlockTile::at(ck_tile::number<1>{}) << std::endl;

        const ck_tile::index_t kGridSize =
            ck_tile::integer_divide_ceil(m, BlockTile::at(ck_tile::number<0>{}));

        using Shape   = ck_tile::AtomicKernelShape<BlockWaves, BlockTile, WaveTile, Vector>;
        using Problem = ck_tile::AtomicKernelProblem<XDataType, Shape>;
        using Kernel  = ck_tile::AtomicKernel<Problem>;

        constexpr ck_tile::index_t kBlockSize  = Config::kBlockSize;
        constexpr ck_tile::index_t kBlockPerCu = 1;

        (void)hipGetLastError(); // clear sticky

        launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                      ck_tile::make_kernel<kBlockPerCu>(
                          Kernel{},
                          kGridSize,
                          kBlockSize,
                          0,
                          static_cast<XDataType*>(x_dev_input.GetDeviceBuffer()),
                          m,
                          n));

        ASSERT_EQ(hipPeekAtLastError(), hipSuccess)
            << "[" << tag << "] hipPeekAtLastError: " << hipGetErrorString(hipGetLastError());
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess)
            << "[" << tag << "] hipDeviceSynchronize failed";

        // host reference computation
        x_dev_input.FromDevice(x_host_dev.mData.data());
        for(int i = 0; i < m; ++i)
            for(int j = 0; j < n; ++j)
                x_host_ref(i, j) = static_cast<XDataType>(1);

        const bool pass = ck_tile::check_err(x_host_dev, x_host_ref);
        EXPECT_TRUE(pass);
    }

    protected:
    // WaveSize = 64 path
    void RunTest(const AtomicKernelParam& params)
    {
        RunTestImpl_<AtomicKernelWaveSize64>(params, /*require_warp_size=*/64, "WS64");
    }

    // WaveSize = 32 path
    void RunTestWave32(const AtomicKernelParam& params)
    {
        RunTestImpl_<AtomicKernelWaveSize32>(params, /*require_warp_size=*/32, "WS32");
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

//
// WaveSize=64 tests (auto-skip on wave32 devices)
//
#if defined(CK_USE_XDL)
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

//
// WaveSize=32 tests (auto-skip on wave64 devices)
//
#else
TEST_P(TestAtomicKernelHalf_1, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelHalf_2, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelHalf_4, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelBF16_1, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelBF16_2, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelBF16_4, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelBF8_1, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelBF8_2, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelFP8_1, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelFP8_2, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelFloat_1, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
TEST_P(TestAtomicKernelFloat_2, TestCorrectnessWS32)
{
    auto [M, N] = GetParam();
    this->RunTestWave32({M, N});
}
#endif

// Common parameter lists
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

#if defined(CK_USE_XDL)
INSTANTIATE_TEST_SUITE_P(TestAtomicKernelSuite,
                         TestAtomicKernelFloat_4,
                         ::testing::Values(std::tuple{64, 8},
                                           std::tuple{64, 16},
                                           std::tuple{64, 32}));
#endif
