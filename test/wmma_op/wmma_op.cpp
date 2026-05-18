// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "test/wmma_op/wmma_op_util.hpp"

static ck::index_t test_case_id = -1;

static ck::index_t case_id = 0;

// gfx12
template <typename SrcType, typename DstType, typename GPUAccType, typename CPUAccType>
bool run_test()
{
    if(!ck::is_gfx12_supported()) // report a warning, but move on.
    {
        fprintf(stderr,
                "----- WARNING: gfx12 not supported, reporting SUCCESS and skipping test -----\n");
        return true;
    }
    else
    {
        fprintf(stderr, "----- INFO: gfx12 supported, running test -----\n");
    }

    using Row         = ck::tensor_layout::gemm::RowMajor;
    using Col         = ck::tensor_layout::gemm::ColumnMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;

    const auto matmul_default   = ck::wmma_op_util::matmul<SrcType, DstType, GPUAccType>;
    const auto matmul_swizzle_a = ck::wmma_op_util::matmul_swizzle_a<SrcType, DstType, GPUAccType>;

    const auto wmma_kernel_container = std::make_tuple(matmul_default, matmul_swizzle_a);
    ck::static_for<0, 2, 1>{}([&](auto i) {
        pass &=
            ck::wmma_op_util::TestWmma<decltype(std::get<ck::Number<i>{}>(wmma_kernel_container)),
                                       SrcType,
                                       SrcType,
                                       DstType,
                                       GPUAccType,
                                       CPUAccType,
                                       decltype(Row{}),
                                       decltype(Col{}),
                                       decltype(Row{}),
                                       PassThrough,
                                       PassThrough,
                                       PassThrough,
                                       1>{}(std::get<ck::Number<i>{}>(wmma_kernel_container));
    });

    return pass ? 1 : 0;
}

// gfx125
template <typename SrcAType,
          typename SrcBType,
          typename DstType,
          typename GPUAccType,
          typename CPUAccType,
          ck::index_t kValue = 1>
bool run_test()
{
    if(!ck::is_gfx125_supported()) // report a warning, but move on.
    {
        fprintf(
            stderr,
            "----- WARNING: gfx1250 not supported, reporting SUCCESS and skipping test -----\n");
        return true;
    }
    else
    {
        fprintf(stderr, "----- INFO: gfx1250 supported, running test -----\n");
    }
    case_id++;

    if(test_case_id != -1 && (test_case_id + 1) != case_id)
    {

        return true;
    }

    using Row         = ck::tensor_layout::gemm::RowMajor;
    using Col         = ck::tensor_layout::gemm::ColumnMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;

    // Pass kValue to both kernels
    const auto matmul_default =
        ck::wmma_op_util::matmul<SrcAType, SrcBType, DstType, GPUAccType, kValue>;
    const auto matmul_swizzle_a =
        ck::wmma_op_util::matmul_swizzle_a<SrcAType, SrcBType, DstType, GPUAccType, kValue>;

    const auto wmma_kernel_container = std::make_tuple(matmul_default, matmul_swizzle_a);

    ck::static_for<0, 2, 1>{}([&](auto i) {
        pass &=
            ck::wmma_op_util::TestWmma<decltype(std::get<ck::Number<i>{}>(wmma_kernel_container)),
                                       SrcAType,
                                       SrcBType,
                                       DstType,
                                       GPUAccType,
                                       CPUAccType,
                                       decltype(Row{}),
                                       decltype(Col{}),
                                       decltype(Row{}),
                                       PassThrough,
                                       PassThrough,
                                       PassThrough,
                                       kValue>{}(std::get<ck::Number<i>{}>(wmma_kernel_container));
    });

    return pass ? 1 : 0;
}

// Individual Google Tests for each run_test invocation
TEST(WMMATest, F32_16x16x32_F16)
{
    auto pass = run_test<ck::half_t, ck::half_t, float, float, float, 32>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x32_F16)
{
    auto pass = run_test<ck::half_t, ck::half_t, ck::half_t, ck::half_t, ck::half_t, 32>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x32_BF16)
{
    auto pass = run_test<ck::bhalf_t, ck::bhalf_t, float, float, float, 32>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, BF16_16x16x32_BF16)
{
    auto pass = run_test<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t, ck::bhalf_t, float, 32>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x64_BF8_BF8)
{
    auto pass = run_test<ck::bf8_t, ck::bf8_t, float, float, float, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x64_BF8_F8)
{
    auto pass = run_test<ck::bf8_t, ck::f8_t, float, float, float, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x64_F8_BF8)
{
    auto pass = run_test<ck::f8_t, ck::bf8_t, float, float, float, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x64_F8_F8)
{
    auto pass = run_test<ck::f8_t, ck::f8_t, float, float, float, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x64_BF8_BF8)
{
    auto pass = run_test<ck::bf8_t, ck::bf8_t, ck::half_t, ck::half_t, ck::half_t, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x64_BF8_F8)
{
    auto pass = run_test<ck::bf8_t, ck::f8_t, ck::half_t, ck::half_t, ck::half_t, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x64_F8_BF8)
{
    auto pass = run_test<ck::f8_t, ck::bf8_t, ck::half_t, ck::half_t, ck::half_t, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x64_F8_F8)
{
    auto pass = run_test<ck::f8_t, ck::f8_t, ck::half_t, ck::half_t, ck::half_t, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x128_BF8_BF8)
{
    auto pass = run_test<ck::bf8_t, ck::bf8_t, float, float, float, 128>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x128_BF8_F8)
{
    auto pass = run_test<ck::bf8_t, ck::f8_t, float, float, float, 128>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x128_F8_BF8)
{
    auto pass = run_test<ck::f8_t, ck::bf8_t, float, float, float, 128>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x128_F8_F8)
{
    auto pass = run_test<ck::f8_t, ck::f8_t, float, float, float, 128>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x128_BF8_BF8)
{
    auto pass = run_test<ck::bf8_t, ck::bf8_t, ck::half_t, ck::half_t, ck::half_t, 128>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x128_BF8_F8)
{
    auto pass = run_test<ck::bf8_t, ck::f8_t, ck::half_t, ck::half_t, ck::half_t, 128>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x128_F8_BF8)
{
    auto pass = run_test<ck::f8_t, ck::bf8_t, ck::half_t, ck::half_t, ck::half_t, 128>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F16_16x16x128_F8_F8)
{
    auto pass = run_test<ck::f8_t, ck::f8_t, ck::half_t, ck::half_t, ck::half_t, 128>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, BF16F32_16x16x32_BF16)
{
    auto pass = run_test<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t, float, float, 32>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, I32_16x16x64_IU8)
{
    auto pass = run_test<int8_t, int8_t, int32_t, int32_t, int32_t, 64>();
    EXPECT_TRUE(pass);
}

TEST(WMMATest, F32_16x16x4_F32)
{
    auto pass = run_test<float, float, float, float, float, 4>();
    EXPECT_TRUE(pass);
}
