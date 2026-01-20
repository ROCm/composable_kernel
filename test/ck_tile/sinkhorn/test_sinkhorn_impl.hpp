// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <cstring>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/sinkhorn_knopp.hpp"
#include "ck_tile/host/kernel_launch.hpp"

template <typename Tuple>
class TestCkTileSinkHorn: public ::testing::Test
{
    protected:
    using XDataType               = std::tuple_element_t<0, Tuple>;
    using ComputeDataType         = std::tuple_element_t<1, Tuple>;
    using YDataType               = std::tuple_element_t<2, Tuple>;
    using BlockWarps_             = std::tuple_element_t<3, Tuple>;
    using BlockTile_              = std::tuple_element_t<4, Tuple>;
    using WarpTile_               = std::tuple_element_t<5, Tuple>;
    using ThreadTile_             = std::tuple_element_t<6, Tuple>;

    using TestSinkhornShape =
        ck_tile::SinkhornKnoppShape<
            BlockWarps_, 
            BlockTile_,
            WarpTile_,
            ThreadTile_
            >;

    void RunGenericTest(const std::vector<ck_tile::index_t>& input_shape, const int max_iterations)
    {

        SinkhornKnoppArgs args{};
        args.input_m       = static_cast<ck_tile::index_t>(input_shape[0]);
        args.max_iterations = max_iterations;

        auto default_stride = {args.input_m, 1};

        ck_tile::HostTensor<XDataType> h_x(input_shape, default_stride);
        ck_tile::HostTensor<YDataType> h_y(input_shape, default_stride);

        ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(h_x);

        auto buffer_size = h_xs.get_element_space_size_in_bytes();
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_y_mem(output_buffer_size);

        args.p_x = static_cast<void*>(d_x_mem.GetDeviceBuffer());
        args.out = static_cast<void*>(d_y_mem.GetDeviceBuffer());

        d_x_mem.ToDevice(h_x.data());
        d_y_mem.ToDevice(h_y.data());

        using Problem = ck_tile::SinkhornKnoppProblem<XDataType,
                                                      YDataType,
                                                      TestSinkhornShape,
                                                      ComputeDataType
                                                     >;
        using Kernel = ck_tile::SinkhornKnoppKernelDummyNonStochastic<
            Problem,
            ck_tile::SinkhornKnoppPolicy>;

        // Launch configuration
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = 1;

        ck_tile::index_t kGridSize = 1 // TODO

        //TODO
        // if(!Kernel::IsSupportedArgument())
        // {
        //     throw std::runtime_error("Wrong! Arguments not supported!\n");
        // }

        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                              kGridSize,
                                              kBlockSize,
                                              0,
                                              args));

        // Reference computation
        // TODO

        // Transfer data from device and check error for each operation
        // TODO

        EXPECT_TRUE(true); // TODO
    } 
};
