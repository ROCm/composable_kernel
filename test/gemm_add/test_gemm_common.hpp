// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/ck.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using I8   = int8_t;
using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

template <typename Tuple>
class TestGemmCommon : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using EDataType   = std::tuple_element_t<3, Tuple>;
    using ALayout     = std::tuple_element_t<4, Tuple>;
    using BLayout     = std::tuple_element_t<5, Tuple>;
    using ELayout     = std::tuple_element_t<6, Tuple>;

    using ProfileCall = bool(*const)(int, int, bool, bool, int, int, int, int, int, int);

    virtual ProfileCall GetImpl() = 0;

    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {16, 32, 64}, {2048, 4096, 8192}, {2048, 1024, 16}};

        bool all_success = true;

        for(auto length : lengths)
        {
            int M        = length[0];
            int N        = length[1];
            int K        = length[2];
            int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB  = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideE  = ck::is_same_v<ELayout, Row> ? N : M;

            all_success =
                all_success &
                GetImpl()(1, 1, false, false, M, N, K, StrideA, StrideB, StrideE);
        }

        EXPECT_TRUE(all_success);
    }
};

template <typename Tuple>
class TestGemmD0Common : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using D0DataType  = std::tuple_element_t<3, Tuple>;
    using EDataType   = std::tuple_element_t<4, Tuple>;
    using ALayout     = std::tuple_element_t<5, Tuple>;
    using BLayout     = std::tuple_element_t<6, Tuple>;
    using D0Layout    = std::tuple_element_t<7, Tuple>;
    using ELayout     = std::tuple_element_t<8, Tuple>;

    using ProfileCall = bool(*const)(int, int, bool, bool, int, int, int, int, int, int, int);

    virtual ProfileCall GetImpl() = 0;

    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {16, 32, 64}, {2048, 4096, 8192}, {2048, 1024, 16}};

        bool all_success = true;

        for(auto length : lengths)
        {
            int M        = length[0];
            int N        = length[1];
            int K        = length[2];
            int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB  = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideD0 = ck::is_same_v<D0Layout, Row> ? N : M;
            int StrideE  = ck::is_same_v<ELayout, Row> ? N : M;

            all_success =
                all_success &
                GetImpl()(1, 1, false, false, M, N, K, StrideA, StrideB, StrideD0, StrideE);
        }

        EXPECT_TRUE(all_success);
    }
};

template <typename Tuple>
class TestGemmD0D1Common : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using D0DataType  = std::tuple_element_t<3, Tuple>;
    using D1DataType  = std::tuple_element_t<4, Tuple>;
    using EDataType   = std::tuple_element_t<5, Tuple>;
    using ALayout     = std::tuple_element_t<6, Tuple>;
    using BLayout     = std::tuple_element_t<7, Tuple>;
    using D0Layout    = std::tuple_element_t<8, Tuple>;
    using D1Layout    = std::tuple_element_t<9, Tuple>;
    using ELayout     = std::tuple_element_t<10, Tuple>;

    using ProfileCall = bool(*const)(int, int, bool, bool, int, int, int, int, int, int, int, int);

    virtual ProfileCall GetImpl() = 0;

    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {16, 32, 64}, {2048, 4096, 8192}, {2048, 1024, 16}};

        bool all_success = true;

        for(auto length : lengths)
        {
            int M        = length[0];
            int N        = length[1];
            int K        = length[2];
            int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
            int StrideB  = ck::is_same_v<BLayout, Row> ? N : K;
            int StrideD0 = ck::is_same_v<D0Layout, Row> ? N : M;
            int StrideD1 = ck::is_same_v<D1Layout, Row> ? N : M;
            int StrideE  = ck::is_same_v<ELayout, Row> ? N : M;

            all_success =
                all_success &
                GetImpl()(1, 1, false, false, M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE);
        }

        EXPECT_TRUE(all_success);
    }
};
