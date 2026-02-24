// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "test_copy.hpp"

struct MemoryCopyParam
{
    MemoryCopyParam(ck_tile::index_t m_, ck_tile::index_t n_, ck_tile::index_t warp_id_)
        : m(m_), n(n_), warp_id(warp_id_)
    {
    }
    ck_tile::index_t m;
    ck_tile::index_t n;
    ck_tile::index_t warp_id;
};

template <typename... Ts>
struct type_list
{
};

template <std::size_t Index, typename List>
struct type_at;

template <std::size_t Index, typename Head, typename... Tail>
struct type_at<Index, type_list<Head, Tail...>> : type_at<Index - 1, type_list<Tail...>>
{
};

template <typename Head, typename... Tail>
struct type_at<0, type_list<Head, Tail...>>
{
    using type = Head;
};

template <typename DataType, bool AsyncCopy = true>
class TestCkTileMemoryCopy : public ::testing::TestWithParam<std::tuple<int, int, int>>
{
    protected:
    void Run(const MemoryCopyParam& memcpy_params)
    {
        using XDataType = DataType;
        using YDataType = DataType;

        ck_tile::index_t m       = memcpy_params.m;
        ck_tile::index_t n       = memcpy_params.n;
        ck_tile::index_t warp_id = memcpy_params.warp_id;

        constexpr auto dword_bytes    = 4;
        const ck_tile::index_t CpyCfg = std::is_same_v<DataType, ck_tile::pk_fp6x16_t> ? 1 : 0;

        ck_tile::HostTensor<XDataType> x_host({m, n});
        ck_tile::HostTensor<YDataType> y_host_dev({m, n});
        ck_tile::HostTensor<int8_t> host_init_buf({x_host.get_element_space_size_in_bytes()});
        std::cout << "input: " << x_host.mDesc << std::endl;
        std::cout << "output: " << y_host_dev.mDesc << std::endl;

        for(size_t i = 0; i < x_host.get_element_space_size_in_bytes(); i++)
            host_init_buf.mData[i] = i % 64;
        memcpy(x_host.mData.data(),
               host_init_buf.mData.data(),
               x_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

        x_buf.ToDevice(x_host.data());

        using BlockTileList = type_list<ck_tile::sequence<64, 8>, ck_tile::sequence<16, 96>>;
        using VectorList    = type_list<ck_tile::sequence<1, dword_bytes / sizeof(DataType)>,
                                        ck_tile::sequence<1, 24>>;
        using BlockWaves    = ck_tile::sequence<2, 1>;
        using BlockTile     = type_at<CpyCfg, BlockTileList>::type;
        using WaveTile      = type_at<CpyCfg, BlockTileList>::type;
        using Vector        = type_at<CpyCfg, VectorList>::type;

        ck_tile::index_t kGridSize =
            ck_tile::integer_divide_ceil(m, BlockTile::at(ck_tile::number<0>{}));

        using Shape   = ck_tile::TileCopyShape<BlockWaves, BlockTile, WaveTile, Vector>;
        using Problem = ck_tile::TileCopyProblem<DataType, Shape, AsyncCopy, CpyCfg>;
        using Kernel  = ck_tile::TileCopy<Problem>;

        constexpr ck_tile::index_t kBlockSize  = 128;
        constexpr ck_tile::index_t kBlockPerCu = 1;
        // when copy fp6x16 buffer, tread it as int8 buffer and recompute n-dim size.
        ck_tile::index_t cpy_n =
            CpyCfg == 1 ? n * sizeof(DataType) /
                              (sizeof(int8_t) * ck_tile::numeric_traits<DataType>::PackedSize)
                        : n;

        auto ms = launch_kernel(
            ck_tile::stream_config{nullptr, true},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                              kGridSize,
                                              kBlockSize,
                                              0,
                                              static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                              static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                              m,
                                              cpy_n,
                                              warp_id));

        auto bytes = 2 * m * n * sizeof(DataType) / ck_tile::numeric_traits<DataType>::PackedSize;
        std::cout << "elapsed: " << ms << " (ms)" << std::endl;
        std::cout << (bytes * 1e-6 / ms) << " (GB/s)" << std::endl;

        // reference
        y_buf.FromDevice(y_host_dev.mData.data());
        bool pass = ck_tile::check_err(y_host_dev, x_host);
        EXPECT_TRUE(pass);
    }
};

class TestCkTileMemoryCopyF6x16Async : public TestCkTileMemoryCopy<ck_tile::pk_fp6x16_t, true>
{
};

class TestCkTileMemoryCopyF6x16 : public TestCkTileMemoryCopy<ck_tile::pk_fp6x16_t, false>
{
};

class TestCkTileMemoryCopyHalfAsync : public TestCkTileMemoryCopy<ck_tile::half_t>
{
};

class TestCkTileMemoryCopyHalfSync : public TestCkTileMemoryCopy<ck_tile::half_t, false>
{
};

class TestCkTileMemoryCopyFloatAsync : public TestCkTileMemoryCopy<float>
{
};

class TestCkTileMemoryCopyFP8Async : public TestCkTileMemoryCopy<ck_tile::fp8_t>
{
};

TEST_P(TestCkTileMemoryCopyF6x16, TestCorrectness)
{
    auto [M, N, warp_id] = GetParam();
    this->Run({M, N, warp_id});
}

TEST_P(TestCkTileMemoryCopyF6x16Async, TestCorrectness)
{
    auto [M, N, warp_id] = GetParam();
    this->Run({M, N, warp_id});
}

TEST_P(TestCkTileMemoryCopyHalfAsync, TestCorrectness)
{
    auto [M, N, warp_id] = GetParam();
    this->Run({M, N, warp_id});
}

TEST_P(TestCkTileMemoryCopyHalfSync, TestCorrectness)
{
    auto [M, N, warp_id] = GetParam();
    this->Run({M, N, warp_id});
}

TEST_P(TestCkTileMemoryCopyFloatAsync, TestCorrectness)
{
    auto [M, N, warp_id] = GetParam();
    this->Run({M, N, warp_id});
}

TEST_P(TestCkTileMemoryCopyFP8Async, TestCorrectness)
{
    auto [M, N, warp_id] = GetParam();
    this->Run({M, N, warp_id});
}

INSTANTIATE_TEST_SUITE_P(TestCkTileMemCopySuite,
                         TestCkTileMemoryCopyF6x16,
                         ::testing::Values(std::tuple{32, 128, 0},
                                           std::tuple{64, 256, 0},
                                           std::tuple{32, 128, 1},
                                           std::tuple{64, 256, 1}));

INSTANTIATE_TEST_SUITE_P(TestCkTileMemCopySuite,
                         TestCkTileMemoryCopyF6x16Async,
                         ::testing::Values(std::tuple{32, 128, 0},
                                           std::tuple{64, 256, 0},
                                           std::tuple{32, 128, 1},
                                           std::tuple{64, 256, 1}));

INSTANTIATE_TEST_SUITE_P(TestCkTileMemCopySuite,
                         TestCkTileMemoryCopyHalfAsync,
                         ::testing::Values(std::tuple{64, 8, 0},
                                           std::tuple{63, 8, 0},
                                           std::tuple{63, 2, 0},
                                           std::tuple{127, 30, 0},
                                           std::tuple{64, 8, 1},
                                           std::tuple{63, 8, 1},
                                           std::tuple{63, 2, 1},
                                           std::tuple{127, 30, 1},
                                           std::tuple{16384, 16384, 0},
                                           std::tuple{16384, 16384, 1}));

INSTANTIATE_TEST_SUITE_P(TestCkTileMemCopySuite,
                         TestCkTileMemoryCopyHalfSync,
                         ::testing::Values(std::tuple{64, 8, 0},
                                           std::tuple{63, 8, 0},
                                           std::tuple{63, 2, 0},
                                           std::tuple{127, 30, 0},
                                           std::tuple{64, 8, 1},
                                           std::tuple{63, 8, 1},
                                           std::tuple{63, 2, 1},
                                           std::tuple{127, 30, 1},
                                           std::tuple{16384, 16384, 0},
                                           std::tuple{16384, 16384, 1}));

INSTANTIATE_TEST_SUITE_P(TestCkTileMemCopySuite,
                         TestCkTileMemoryCopyFloatAsync,
                         ::testing::Values(std::tuple{64, 8, 0},
                                           std::tuple{63, 8, 0},
                                           std::tuple{63, 2, 0},
                                           std::tuple{127, 30, 0},
                                           std::tuple{64, 8, 1},
                                           std::tuple{63, 8, 1},
                                           std::tuple{63, 2, 1},
                                           std::tuple{127, 30, 1},
                                           std::tuple{16384, 16384, 0},
                                           std::tuple{16384, 16384, 1}));

INSTANTIATE_TEST_SUITE_P(TestCkTileMemCopySuite,
                         TestCkTileMemoryCopyFP8Async,
                         ::testing::Values(std::tuple{64, 8, 0},
                                           std::tuple{63, 8, 0},
                                           std::tuple{63, 4, 0},
                                           std::tuple{127, 20, 0},
                                           std::tuple{64, 8, 1},
                                           std::tuple{63, 8, 1},
                                           std::tuple{63, 4, 1},
                                           std::tuple{127, 20, 1},
                                           std::tuple{16384, 16384, 0},
                                           std::tuple{16384, 16384, 1}));
