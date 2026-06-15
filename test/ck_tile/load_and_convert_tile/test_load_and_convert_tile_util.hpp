// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/common.hpp"
#include "print_matrix.hpp"
#include "kernel.hpp"

// Enum struct specifying what kind of test matrix to use
enum struct TestMatrixType
{
    MonotonicSequence   = 0,
    UniformDistribution = 1
};

static constexpr auto matrix_type    = TestMatrixType::UniformDistribution;
static constexpr bool kPrintMatrices = false;

template <typename Tuple>
class TestLoadAndConvert : public ::testing::Test
{
    public:
    using XDataType     = std::tuple_element_t<0, Tuple>;
    using YDataType     = std::tuple_element_t<1, Tuple>;
    using LoadTranspose = std::tuple_element_t<2, Tuple>;

    protected:
    void RunTest()
    {
        constexpr ck_tile::index_t M = 256;
        constexpr ck_tile::index_t N = 256;

        ck_tile::HostTensor<XDataType> h_a({M, N});
        ck_tile::HostTensor<YDataType> h_c({M, N});

        if constexpr(matrix_type == TestMatrixType::MonotonicSequence)
        {
            ck_tile::HostTensor<float> h_a_tmp({M, N});
            ck_tile::FillMonotonicSeq<float>{0.0, 0.1}(h_a_tmp);
            ck_tile::reference_unary_elementwise<float, XDataType, float>(
                h_a_tmp, h_a, [](const auto& x) { return x; });
        }
        else
        {
            ck_tile::FillUniformDistributionIntegerValue<XDataType>{-5.0, 5.0}(h_a);
        }

        ck_tile::DeviceMem d_a(h_a.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_c(h_c.get_element_space_size_in_bytes());

        d_a.ToDevice(h_a.data());

        using BlockWarps = ck_tile::sequence<4, 4>;
        using BlockTile  = ck_tile::sequence<512, 32>;
        using WarpTile   = ck_tile::sequence<64, 8>;
        using Vector     = ck_tile::sequence<1, 8>;

        using Shape   = ck_tile::LoadAndConvertShape<BlockWarps, BlockTile, WarpTile, Vector>;
        using Problem = ck_tile::LoadAndConvertProblem<XDataType, YDataType, Shape, LoadTranspose>;
        using Policy  = ck_tile::LoadAndConvertPolicy<Problem>;
        using Kernel  = ck_tile::LoadAndConvertKernel<Problem, Policy>;

        const ck_tile::index_t block_size = Kernel::BlockSize();
        const ck_tile::index_t grid_size  = ck_tile::integer_divide_ceil(M, Shape::Block_M);

        launch_kernel(ck_tile::stream_config{nullptr, true},
                      make_kernel<1>(Kernel{},
                                     dim3(grid_size),
                                     dim3(block_size),
                                     0,
                                     static_cast<const XDataType*>(d_a.GetDeviceBuffer()),
                                     static_cast<YDataType*>(d_c.GetDeviceBuffer()),
                                     M,
                                     N));

        ck_tile::hip_check_error(hipDeviceSynchronize());
        d_c.FromDevice(h_c.data());
        ck_tile::HostTensor<YDataType> h_a_ref({M, N});
        ck_tile::reference_unary_elementwise<XDataType, YDataType, float>(
            h_a, h_a_ref, [](const auto& x) { return x; });
        bool pass = ck_tile::check_err(h_c, h_a_ref);

        if constexpr(kPrintMatrices)
        {
            auto [width, precision] = matrix_type == TestMatrixType::MonotonicSequence
                                          ? std::make_pair(3, 3)
                                          : std::make_pair(2, 6);
            print_matrix(h_a, "Matrix A", width, precision);
            print_matrix(h_c, "Matrix C", width, precision);
        }

        EXPECT_TRUE(pass);
    }
};
