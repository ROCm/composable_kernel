// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <tuple>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"

template <typename Params>
class TestCkTileMultiReduceOutputTails : public ::testing::Test
{
    protected:
    using Y                     = std::tuple_element_t<0, Params>;
    static constexpr bool Multi = std::tuple_element_t<1, Params>::value;

    void Run(int rows)
    {
        using namespace ck_tile;
        constexpr index_t cols = 512;
        using Shape =
            Reduce2dShape<sequence<4, 1>, sequence<128, 128>, sequence<32, 128>, sequence<8, 8>>;
        using Ops     = tuple<ReduceOp::Add>;
        using Kept    = sequence<0>;
        using Reduced = sequence<1>;
        using Problem = Reduce2dProblem<half_t, float, Y, Shape, Ops, Kept, Reduced, 2>;
        using Kernel  = MultiReduce2d<Problem, Reduce2dDefaultPolicy, Multi>;
        auto shape    = make_tuple(rows, cols);
        auto strides  = make_tuple(cols, 1);
        if constexpr(Multi && sizeof(Y) < sizeof(float))
        {
            if(rows % 8 != 0)
            {
                // Packed half/BF16 atomics cannot update an incomplete output vector safely.
                EXPECT_FALSE(Kernel::IsSupportedArgument(cols, strides, rows));
                return;
            }
        }
        ASSERT_TRUE(Kernel::IsSupportedArgument(cols, strides, rows));

        // Exact sums isolate boundary errors from low-precision accumulation error.
        std::vector<half_t> input(rows * cols, type_convert<half_t>(1.0f));
        const Y guard = type_convert<Y>(Multi ? -0.0f : 42.0f);
        std::vector<Y> output(rows + Shape::Block_M, guard);
        std::fill_n(output.begin(), rows, type_convert<Y>(0.0f));
        DeviceMem x(input.size() * sizeof(half_t));
        DeviceMem y(output.size() * sizeof(Y));
        x.ToDevice(input.data());
        y.ToDevice(output.data());

        const auto group_size =
            typename Kernel::TilePartitioner{cols}.GetBlockGroupParams().at(number<1>{});
        const index_t blocks = integer_divide_ceil(rows, Shape::Block_M) * (Multi ? group_size : 1);
        const auto pass      = make_tuple(element_wise::PassThrough{});
        launch_kernel(stream_config{nullptr, false, 0},
                      make_kernel<1>(Kernel{},
                                     dim3(blocks),
                                     dim3(Kernel::BlockSize()),
                                     0,
                                     static_cast<const half_t*>(x.GetDeviceBuffer()),
                                     static_cast<Y*>(y.GetDeviceBuffer()),
                                     shape,
                                     strides,
                                     Kept{},
                                     Reduced{},
                                     rows,
                                     pass,
                                     pass,
                                     Ops{}));
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        y.FromDevice(output.data());

        int touched = 0;
        for(int i = 0; i < rows; ++i)
            EXPECT_EQ(type_convert<double>(output[i]), cols) << "row=" << i;
        for(int i = rows; i < static_cast<int>(output.size()); ++i)
            touched += std::memcmp(&output[i], &guard, sizeof(Y)) != 0;
        EXPECT_EQ(touched, 0);
    }
};

template <typename T, bool Multi>
using TailCase = std::tuple<T, std::bool_constant<Multi>>;
using Cases    = ::testing::Types<TailCase<float, false>,
                                  TailCase<ck_tile::half_t, false>,
                                  TailCase<ck_tile::bf16_t, false>,
                                  TailCase<double, false>,
                                  TailCase<float, true>,
                                  TailCase<ck_tile::half_t, true>,
                                  TailCase<ck_tile::bf16_t, true>,
                                  TailCase<double, true>>;
TYPED_TEST_SUITE(TestCkTileMultiReduceOutputTails, Cases);
// Exercise odd and even vector tails in the first and second block, plus an aligned control.
TYPED_TEST(TestCkTileMultiReduceOutputTails, Rows1) { this->Run(1); }
TYPED_TEST(TestCkTileMultiReduceOutputTails, Rows65) { this->Run(65); }
TYPED_TEST(TestCkTileMultiReduceOutputTails, Rows193) { this->Run(193); }
TYPED_TEST(TestCkTileMultiReduceOutputTails, Rows194) { this->Run(194); }
TYPED_TEST(TestCkTileMultiReduceOutputTails, Rows192AlignedControl) { this->Run(192); }
