// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <getopt.h>

#include "ck/library/utility/host_common_util.hpp"
#include "profiler/profile_reduce_impl.hpp"
#include <gtest/gtest.h>
using namespace ck;

struct ReduceParam
{
    bool do_verification{true};
    bool propagateNan{false};
    bool useIndex{true};
    bool time_kernel{false};
    bool do_dumpout{false};
    int init_method{2};
    float alpha{1.0f};
    float beta{0.0f};
    std::vector<size_t> inLengths{64, 4, 280, 82};
    std::vector<int> reduceDims{0, 1, 2, 3};
    ReduceTensorOp reduceOpId{ReduceTensorOp::AMAX};
};

std::vector<std::vector<int>> settGenericReduceDim()
{
    return {{0, 1, 2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}, {0}, {1}, {2}, {3}};
}

template <typename T>
class ReduceWithIndexTest : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, T>;
    using AccDataType = std::tuple_element_t<1, T>;
    using OutDataType = std::tuple_element_t<2, T>;

    static std::vector<ReduceParam> params;

    static void SetUpTestSuite()
    {
        // set testcase variables
        ReduceParam _sett0;
        // set testcase variables
        const auto settReduceDim = settGenericReduceDim();

        for(std::size_t i(0); i < settReduceDim.size(); ++i)
        {
            _sett0.reduceDims = settReduceDim[i];
            _sett0.reduceOpId = ReduceTensorOp::AMAX;
            params.emplace_back(_sett0);
            _sett0.reduceOpId = ReduceTensorOp::MIN;
            params.push_back(_sett0);
            _sett0.reduceOpId = ReduceTensorOp::MAX;
            params.push_back(_sett0);
        }
    };

    void Run()
    {
        for(auto param : this->params)
        {
            bool success = ck::profiler::profile_reduce_impl<InDataType, AccDataType, OutDataType>(
                param.do_verification,
                param.init_method,
                param.do_dumpout,
                param.time_kernel,
                param.inLengths,
                param.reduceDims,
                param.reduceOpId,
                param.propagateNan,
                param.useIndex,
                param.alpha,
                param.beta);
            // EXPECT_TRUE
            EXPECT_TRUE(success);
        }
    }
};

template <typename T>
std::vector<ReduceParam> ReduceWithIndexTest<T>::params = {};

using Reduce_float_types       = ::testing::Types<std::tuple<float, float, float>>;
using Reduce_double_types      = ::testing::Types<std::tuple<double, double, double>>;
using Reduce_int8t_types       = ::testing::Types<std::tuple<int8_t, int8_t, int8_t>>;
using Reduce_half_types        = ::testing::Types<std::tuple<ck::half_t, ck::half_t, ck::half_t>>;
using Reduce_bhalf_float_Types = ::testing::Types<std::tuple<ck::bhalf_t, float, ck::bhalf_t>>;

template <typename TType>
class ReduceWithIndexFloat : public ReduceWithIndexTest<TType>
{
};

template <typename TType>
class ReduceWithIndexDouble : public ReduceWithIndexTest<TType>
{
};

template <typename TType>
class ReduceWithIndexInt8 : public ReduceWithIndexTest<TType>
{
};

template <typename TType>
class ReduceWithIndexHalf : public ReduceWithIndexTest<TType>
{
};

template <typename TType>
class ReduceWithIndexBHalfFloat : public ReduceWithIndexTest<TType>
{
};

TYPED_TEST_SUITE(ReduceWithIndexFloat, Reduce_float_types);
TYPED_TEST_SUITE(ReduceWithIndexDouble, Reduce_double_types);
TYPED_TEST_SUITE(ReduceWithIndexInt8, Reduce_int8t_types);
TYPED_TEST_SUITE(ReduceWithIndexHalf, Reduce_half_types);
TYPED_TEST_SUITE(ReduceWithIndexBHalfFloat, Reduce_bhalf_float_Types);

TYPED_TEST(ReduceWithIndexFloat, ReduceWithIndexTestFloat)
{
    // trigger Run() -> Generic
    this->Run();
}

TYPED_TEST(ReduceWithIndexDouble, ReduceWithIndexTestDouble)
{
    // trigger Run() -> Generic
    this->Run();
}

TYPED_TEST(ReduceWithIndexInt8, ReduceWithIndexTestInt8)
{
    // trigger Run() -> Generic
    this->Run();
}

TYPED_TEST(ReduceWithIndexHalf, ReduceWithIndexTestHalf)
{
    // trigger Run() -> Generic
    this->Run();
}

TYPED_TEST(ReduceWithIndexBHalfFloat, ReduceWithIndexTestBHalfFloat)
{
    // trigger Run() -> Generic
    this->Run();
}
