// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <getopt.h>

#include "ck/library/utility/host_common_util.hpp"
#include "profiler/profile_reduce_impl.hpp"
#include <gtest/gtest.h>
using namespace ck;

struct ReduceParam
{
    bool do_verification;
    bool propagateNan;
    bool useIndex;
    bool time_kernel;
    bool do_dumpout;
    int init_method;
    float alpha;
    float beta;
    std::vector<size_t> inLengths;
    std::vector<int> reduceDims;
    ReduceTensorOp reduceOpId;
};

std::vector<std::vector<int>> settGenericReduceDim()
{
    return {{0, 1, 2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}, {0}, {1}, {2}, {3}};
}

ReduceParam sett0()
{
    return {/*do_verification*/ true,
            /*propagateNan*/ false,
            /*useIndex*/ false,
            /*time_kernel*/ false,
            /*do_dumpout*/ false,
            /*init_method*/ 2,
            /*alpha*/ 1.0f,
            /*beta*/ 0.0f,
            /*inLengths*/ {64, 4, 280, 82},
            /*reduceDims*/ {0, 1, 2, 3},
            /*reduceOpId*/ ReduceTensorOp::AMAX};
}

template <typename T>
class ReduceNoIndexTest : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, T>;
    using AccDataType = std::tuple_element_t<1, T>;
    using OutDataType = std::tuple_element_t<2, T>;

    static std::vector<ReduceParam> params;

    static void SetUpTestSuite()
    {
        // set testcase variables
        ReduceParam _sett0 = sett0();
        // + reduce dims: Generic;
        // set testcase variables
        const auto settReduceDim = settGenericReduceDim();

        for(std::size_t i(0); i < settReduceDim.size(); ++i)
        {
            _sett0.reduceOpId = ReduceTensorOp::AMAX;
            _sett0.reduceDims = settReduceDim[i];
            params.push_back(_sett0);
            _sett0.reduceOpId = ReduceTensorOp::MIN;
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
std::vector<ReduceParam> ReduceNoIndexTest<T>::params = {};

using Reduce_float_types       = ::testing::Types<std::tuple<float, float, float>>;
using Reduce_double_types      = ::testing::Types<std::tuple<double, double, double>>;
using Reduce_int8t_types       = ::testing::Types<std::tuple<int8_t, int8_t, int8_t>>;
using Reduce_half_types        = ::testing::Types<std::tuple<ck::half_t, ck::half_t, ck::half_t>>;
using Reduce_bhalf_float_Types = ::testing::Types<std::tuple<ck::bhalf_t, float, ck::bhalf_t>>;

template <typename TType>
class ReduceNoIndexFloat : public ReduceNoIndexTest<TType>
{
};

template <typename TType>
class ReduceNoIndexDouble : public ReduceNoIndexTest<TType>
{
};

template <typename TType>
class ReduceNoIndexInt8 : public ReduceNoIndexTest<TType>
{
};

template <typename TType>
class ReduceNoIndexHalf : public ReduceNoIndexTest<TType>
{
};

template <typename TType>
class ReduceNoIndexBHalfFloat : public ReduceNoIndexTest<TType>
{
};

TYPED_TEST_SUITE(ReduceNoIndexFloat, Reduce_float_types);
TYPED_TEST_SUITE(ReduceNoIndexDouble, Reduce_double_types);
TYPED_TEST_SUITE(ReduceNoIndexInt8, Reduce_int8t_types);
TYPED_TEST_SUITE(ReduceNoIndexHalf, Reduce_half_types);
TYPED_TEST_SUITE(ReduceNoIndexBHalfFloat, Reduce_bhalf_float_Types);

TYPED_TEST(ReduceNoIndexFloat, ReduceNoIndexTestFloat)
{
    // trigger Run() -> Generic
    this->Run();
}

TYPED_TEST(ReduceNoIndexDouble, ReduceNoIndexTestDouble)
{
    // trigger Run() -> Generic
    this->Run();
}

TYPED_TEST(ReduceNoIndexInt8, ReduceNoIndexTestInt8)
{
    // trigger Run() -> Generic
    this->Run();
}

TYPED_TEST(ReduceNoIndexHalf, ReduceNoIndexTestHalf)
{
    // trigger Run() -> Generic
    this->Run();
}

TYPED_TEST(ReduceNoIndexBHalfFloat, ReduceNoIndexTestBHalfFloat)
{
    // trigger Run() -> Generic
    this->Run();
}
