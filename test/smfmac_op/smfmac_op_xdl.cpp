// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "test/smfmac_op/smfmac_op_util.hpp"

using BF16        = ck::bhalf_t;
using F16         = ck::half_t;
using F32         = float;
using Row         = ck::tensor_layout::gemm::RowMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename Tuple>
class TestSmfmac : public ::testing::Test
{
    protected:
    using Src1Type    = std::tuple_element_t<0, Tuple>;
    using Src1VecSize = std::tuple_element_t<1, Tuple>;
    using Src2Type    = std::tuple_element_t<2, Tuple>;
    using Src2VecSize = std::tuple_element_t<3, Tuple>;
    using DstType     = std::tuple_element_t<4, Tuple>;
    using AccVecSize  = std::tuple_element_t<5, Tuple>;
    using GPUAccType  = std::tuple_element_t<6, Tuple>;
    using CPUAccType  = std::tuple_element_t<7, Tuple>;
    using M           = std::tuple_element_t<8, Tuple>;
    using N           = std::tuple_element_t<9, Tuple>;
    using K           = std::tuple_element_t<10, Tuple>;

    void Run()
    {
        bool pass                     = true;
        constexpr auto matmul_default = ck::smfmac_op_util::matmul<Src1Type,
                                                                   Src1VecSize::value,
                                                                   Src2Type,
                                                                   Src2VecSize::value,
                                                                   GPUAccType,
                                                                   AccVecSize::value,
                                                                   DstType,
                                                                   M::value,
                                                                   N::value,
                                                                   K::value>;

        constexpr auto smfmac_kernel_container = std::make_tuple(matmul_default);

        ck::static_for<0, std::tuple_size_v<decltype(smfmac_kernel_container)>, 1>{}([&](auto i) {
            pass &= ck::smfmac_op_util::TestSmfmac<
                std::tuple_element_t<i.value, decltype(smfmac_kernel_container)>,
                Src1Type,
                Src2Type,
                DstType,
                GPUAccType,
                CPUAccType,
                decltype(Row{}),
                decltype(Row{}),
                decltype(Row{}),
                PassThrough,
                PassThrough,
                PassThrough,
                AccVecSize::value,
                M::value,
                N::value,
                K::value>{}(std::get<ck::Number<i>{}>(smfmac_kernel_container));
        });

        EXPECT_TRUE(pass);
    }
};

using four_t      = std::integral_constant<ck::index_t, 4>;
using eight_t     = std::integral_constant<ck::index_t, 8>;
using sixteen_t   = std::integral_constant<ck::index_t, 16>;
using thirtytwo_t = std::integral_constant<ck::index_t, 32>;

using KernelTypes = ::testing::Types<
    std::tuple<F16, four_t, F16, eight_t, F32, four_t, F32, F32, sixteen_t, sixteen_t, thirtytwo_t>,
    std::tuple<BF16,
               four_t,
               BF16,
               eight_t,
               F32,
               four_t,
               F32,
               F32,
               sixteen_t,
               sixteen_t,
               thirtytwo_t>,
    std::tuple<F16,
               four_t,
               F16,
               eight_t,
               F32,
               sixteen_t,
               F32,
               F32,
               thirtytwo_t,
               thirtytwo_t,
               sixteen_t>,
    std::tuple<BF16,
               four_t,
               BF16,
               eight_t,
               F32,
               sixteen_t,
               F32,
               F32,
               thirtytwo_t,
               thirtytwo_t,
               sixteen_t>>;

TYPED_TEST_SUITE(TestSmfmac, KernelTypes);
TYPED_TEST(TestSmfmac, TestSmfmacFP16BF16) { this->Run(); }
