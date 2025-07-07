// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/reduce/kernel/reduce2d_kernel.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_problem.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_default_policy.hpp"
#include "ck_tile/ops/reduce/pipeline/reduce2d_shape.hpp"
#include "ck_tile/host/reference/reference_reduce.hpp"

template <typename Tuple>
class TestCkTileReduce2d : public ::testing::Test
{
    protected:
    using XDataType       = std::tuple_element_t<0, Tuple>;
    using ComputeDataType = std::tuple_element_t<1, Tuple>;
    using YDataType       = std::tuple_element_t<2, Tuple>;
    using ReduceOpType    = std::tuple_element_t<3, Tuple>;
    using BlockWarps_     = std::tuple_element_t<4, Tuple>;
    using BlockTile_      = std::tuple_element_t<5, Tuple>;
    using WarpTile_       = std::tuple_element_t<6, Tuple>;
    using Vector_         = std::tuple_element_t<7, Tuple>;

    using TestReduce2dShape = ck_tile::Reduce2dShape<BlockWarps_, BlockTile_, WarpTile_, Vector_>;

    void RunTest(ck_tile::index_t m, ck_tile::index_t n, ck_tile::index_t k)
    {
        // Problem shape: 3D tensor [M, N, K] -> reduce along [N, K] -> output [M]
        std::vector<ck_tile::index_t> problem_shape = {m, n, k};
        std::vector<ck_tile::index_t> strides(3);
        strides[0] = n * k; // M stride
        strides[1] = k;     // N stride
        strides[2] = 1;     // K stride

        constexpr auto kept_dim    = ck_tile::sequence<0>{};
        constexpr auto reduce_dims = ck_tile::sequence<1, 2>{};

        ck_tile::HostTensor<XDataType> h_x(problem_shape, strides);
        ck_tile::HostTensor<YDataType> h_y({problem_shape[kept_dim.at(0)]}, {1});
        ck_tile::HostTensor<YDataType> h_y_ref({problem_shape[kept_dim.at(0)]}, {1});

        ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(h_x);
        h_y.SetZero();
        h_y_ref.SetZero();

        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_y_mem(h_y.get_element_space_size_in_bytes());

        d_x_mem.ToDevice(h_x.data());
        d_y_mem.ToDevice(h_y.data()); // Initialize device output buffer

        // Problem and kernel setup
        using Problem = ck_tile::
            Reduce2dProblem<XDataType, ComputeDataType, YDataType, TestReduce2dShape, ReduceOpType>;

        using Kernel = ck_tile::Reduce<Problem>;

        // Launch configuration
        constexpr ck_tile::index_t kBlockSize  = 256;
        constexpr ck_tile::index_t kBlockPerCu = 1;
        ck_tile::index_t kGridSize =
            (problem_shape[kept_dim.at(0)] + TestReduce2dShape::Block_M - 1) /
            TestReduce2dShape::Block_M;

        auto input_shape =
            ck_tile::make_tuple(problem_shape[0], problem_shape[1], problem_shape[2]);
        auto input_strides = ck_tile::make_tuple(strides[0], strides[1], strides[2]);

        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0},
                               ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                   Kernel{},
                                   kGridSize,
                                   kBlockSize,
                                   0,
                                   static_cast<XDataType*>(d_x_mem.GetDeviceBuffer()),
                                   static_cast<YDataType*>(d_y_mem.GetDeviceBuffer()),
                                   input_shape,
                                   input_strides,
                                   kept_dim,
                                   reduce_dims));

        // Get results back
        d_y_mem.FromDevice(h_y.data());

        // Reference computation
        ck_tile::reference_reduce<XDataType, ComputeDataType, YDataType>(
            h_x, h_y_ref, ReduceOpType{}, kept_dim, reduce_dims);

        // Calculate proper error thresholds based on data types and number of accumulations
        const auto total_reduce_elements = n * k;
        const auto rtol = ck_tile::get_relative_threshold<XDataType, YDataType, ComputeDataType>(
            total_reduce_elements);
        const auto atol = ck_tile::get_absolute_threshold<XDataType, YDataType, ComputeDataType>(
            5.0f, total_reduce_elements);

        bool result =
            ck_tile::check_err(h_y, h_y_ref, "Error: Incorrect reduce results!", rtol, atol);
        EXPECT_TRUE(result);
    }

    void RunTest2D(ck_tile::index_t m, ck_tile::index_t n)
    {
        // 2D case: [M, N] -> reduce along [N] -> output [M]
        RunTest(m, n, 1);
    }
};

// Shape parameters for different test configurations
using Shape1_BlockWarps = ck_tile::sequence<4, 1>;
using Shape1_BlockTile  = ck_tile::sequence<128, 128>;
using Shape1_WarpTile   = ck_tile::sequence<32, 128>;
using Shape1_Vector     = ck_tile::sequence<8, 8>;

using Shape2_BlockWarps = ck_tile::sequence<2, 2>; // Cross-warp reduction test
using Shape2_BlockTile  = ck_tile::sequence<2, 1024>;
using Shape2_WarpTile   = ck_tile::sequence<1, 512>;
using Shape2_Vector     = ck_tile::sequence<1, 8>;

// Test configurations for different data types and operations
using TestConfig_F32_Add = std::tuple<float,
                                      float,
                                      float,
                                      ck_tile::ReduceOp::Add,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_Vector>;

using TestConfig_F16_Add = std::tuple<ck_tile::half_t,
                                      float,
                                      ck_tile::half_t,
                                      ck_tile::ReduceOp::Add,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_Vector>;

using TestConfig_F32_CrossWarp = std::tuple<float,
                                            float,
                                            float,
                                            ck_tile::ReduceOp::Add,
                                            Shape2_BlockWarps,
                                            Shape2_BlockTile,
                                            Shape2_WarpTile,
                                            Shape2_Vector>;

using TestConfig_F32_Max = std::tuple<float,
                                      float,
                                      float,
                                      ck_tile::ReduceOp::Max,
                                      Shape1_BlockWarps,
                                      Shape1_BlockTile,
                                      Shape1_WarpTile,
                                      Shape1_Vector>;

using TestConfig_F32_SquareAdd = std::tuple<float,
                                            float,
                                            float,
                                            ck_tile::ReduceOp::SquareAdd,
                                            Shape1_BlockWarps,
                                            Shape1_BlockTile,
                                            Shape1_WarpTile,
                                            Shape1_Vector>;

using TestTypes = ::testing::Types<TestConfig_F32_Add,
                                   TestConfig_F16_Add,
                                   TestConfig_F32_CrossWarp,
                                   TestConfig_F32_Max,
                                   TestConfig_F32_SquareAdd>;

TYPED_TEST_SUITE(TestCkTileReduce2d, TestTypes);

TYPED_TEST(TestCkTileReduce2d, test) { this->RunTest(128, 128, 1); }

TYPED_TEST(TestCkTileReduce2d, Reduce3D_512_1024_16) { this->RunTest(512, 1024, 16); }

TYPED_TEST(TestCkTileReduce2d, Reduce3D_150_170_6) { this->RunTest(150, 64, 3); }

TYPED_TEST(TestCkTileReduce2d, Reduce2D_128_128) { this->RunTest2D(128, 128); }
