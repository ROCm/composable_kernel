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
class TestCkTileSinkHorn : public ::testing::Test
{
    protected:
    using XDataType       = std::tuple_element_t<0, Tuple>;
    using ComputeDataType = std::tuple_element_t<1, Tuple>;
    using YDataType       = std::tuple_element_t<2, Tuple>;
    using BlockWarps_     = std::tuple_element_t<3, Tuple>;
    using BlockTile_      = std::tuple_element_t<4, Tuple>;
    using WarpTile_       = std::tuple_element_t<5, Tuple>;
    using ThreadTile_     = std::tuple_element_t<6, Tuple>;

    using TestSinkhornShape =
        ck_tile::SinkhornKnoppShape<BlockWarps_, BlockTile_, WarpTile_, ThreadTile_>;

    void RunGenericTest(const std::vector<ck_tile::index_t>& input_shape, const int max_iterations)
    {
        auto input_n        = input_shape[0];
        auto default_stride = {input_n, 1};

        ck_tile::HostTensor<XDataType> h_x(input_shape, default_stride);
        ck_tile::HostTensor<YDataType> h_y(input_shape, default_stride);

        ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(h_x);

        auto buffer_size = h_x.get_element_space_size_in_bytes();
        ck_tile::DeviceMem d_x_mem(h_x.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_y_mem(buffer_size);

        ck_tile::SinkhornKnoppArgs args{static_cast<void*>(d_y_mem.GetDeviceBuffer()),
                                        static_cast<void*>(d_x_mem.GetDeviceBuffer()),
                                        input_n,
                                        max_iterations};

        d_x_mem.ToDevice(h_x.data());
        d_y_mem.ToDevice(h_y.data());

        using Problem =
            ck_tile::SinkhornKnoppProblem<XDataType, YDataType, TestSinkhornShape, ComputeDataType>;
        using Kernel =
            ck_tile::SinkhornKnoppKernelDummyNonStochastic<Problem,
                                                           ck_tile::SinkhornKnoppDefaultPolicy>;

        // Launch configuration
        const ck_tile::index_t kBlockSize      = Kernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = 1;

        ck_tile::index_t kGridSize = 1; // TODO

        // TODO
        //  if(!Kernel::IsSupportedArgument())
        //  {
        //      throw std::runtime_error("Wrong! Arguments not supported!\n");
        //  }

        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr, false, 0},
            ck_tile::make_kernel<kBlockPerCu>(Kernel{}, kGridSize, kBlockSize, 0, args));

        // Reference computation
        ck_tile::HostTensor<YDataType> h_y_ref(input_shape, default_stride);
        sinkhorn_knopp_ref<XDataType, ComputeDataType, YDataType>(h_x, h_y_ref, max_iterations);

        // TODO: Test whether or not output is actually doubly stochastic

        // TODO: Refine tolerances
        const float rtol = 1e-7;
        const float atol = 1e-8;

        // Transfer data from device and check that it matches reference
        d_y_mem.FromDevice(h_y.data());
        bool result = true;
        result &= ck_tile::check_err(
            h_y, h_y_ref, "Error: Sinkhorn-Knopp doesn't match CPU reference!", rtol, atol);

        EXPECT_TRUE(result);
    }
};
