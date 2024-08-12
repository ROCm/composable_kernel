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
    bool useIndex{false};
    bool time_kernel{false};
    bool do_dumpout{false};
    int init_method{2};
    float alpha{1.0f};
    float beta{0.0f};
    std::vector<size_t> inLengths{64, 4, 280, 82};
    std::vector<int> reduceDims{0, 1, 2, 3};
};

const std::vector<std::vector<int>> SetGenericReduceDim()
{
    return {{0, 1, 2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}, {0}, {1}, {2}, {3}};
}

const std::vector<std::vector<std::size_t>> SetGenericInputDim() { return {{64, 4, 280, 82}}; }

const std::vector<std::vector<std::size_t>> Set12DInputDim()
{
    return {{64, 4, 280, 82, 1, 1, 1, 1, 1, 1, 1, 1}, {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}};
}

template <typename T>
class ReduceNoIndexTest : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, T>;
    using AccDataType = std::tuple_element_t<1, T>;
    using OutDataType = std::tuple_element_t<2, T>;

    static std::vector<ReduceParam> params;

    template <typename TFReduce, typename TFInput>
    static void SetInputAndReduceDims(TFReduce&& funSetReduce, TFInput&& funSetInput)
    {
        ReduceParam set;
        const auto setReduceDim = funSetReduce();
        const auto setInputDim  = funSetInput();
        const int N             = setInputDim.size();
        for(std::size_t i(0); i < setReduceDim.size(); ++i)
        {
            set.reduceDims = setReduceDim[i];
            set.inLengths  = setInputDim[i % N];
            params.emplace_back(set);
        }
    }

    static void SetUpTestCase()
    {
        // set testcase variables
        SetInputAndReduceDims(SetGenericReduceDim, SetGenericInputDim);
    }

    template <ReduceTensorOp ReduceOpIdType>
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
                ReduceOpIdType,
                param.propagateNan,
                param.useIndex,
                param.alpha,
                param.beta);
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

template <typename TType>
class ReduceNoIndexFloat12D : public ReduceNoIndexTest<TType>
{
    protected:
    static void SetUpTestCase()
    {
        ReduceNoIndexTest<TType>::SetInputAndReduceDims(SetGenericReduceDim, Set12DInputDim);
    }
};

template <typename TType>
class ReduceNoIndexDouble12D : public ReduceNoIndexTest<TType>
{
    protected:
    static void SetUpTestCase()
    {
        ReduceNoIndexTest<TType>::SetInputAndReduceDims(SetGenericReduceDim, Set12DInputDim);
    }
};

template <typename TType>
class ReduceNoIndexInt812D : public ReduceNoIndexTest<TType>
{
    protected:
    static void SetUpTestCase()
    {
        ReduceNoIndexTest<TType>::SetInputAndReduceDims(SetGenericReduceDim, Set12DInputDim);
    }
};

template <typename TType>
class ReduceNoIndexHalf12D : public ReduceNoIndexTest<TType>
{
    protected:
    static void SetUpTestCase()
    {
        ReduceNoIndexTest<TType>::SetInputAndReduceDims(SetGenericReduceDim, Set12DInputDim);
    }
};

template <typename TType>
class ReduceNoIndexBHalfFloat12D : public ReduceNoIndexTest<TType>
{
    protected:
    static void SetUpTestCase()
    {
        ReduceNoIndexTest<TType>::SetInputAndReduceDims(SetGenericReduceDim, Set12DInputDim);
    }
};

TYPED_TEST_SUITE(ReduceNoIndexFloat, Reduce_float_types);
TYPED_TEST_SUITE(ReduceNoIndexDouble, Reduce_double_types);
TYPED_TEST_SUITE(ReduceNoIndexInt8, Reduce_int8t_types);
TYPED_TEST_SUITE(ReduceNoIndexHalf, Reduce_half_types);
TYPED_TEST_SUITE(ReduceNoIndexBHalfFloat, Reduce_bhalf_float_Types);

TYPED_TEST_SUITE(ReduceNoIndexFloat12D, Reduce_float_types);
TYPED_TEST_SUITE(ReduceNoIndexDouble12D, Reduce_double_types);
TYPED_TEST_SUITE(ReduceNoIndexInt812D, Reduce_int8t_types);
TYPED_TEST_SUITE(ReduceNoIndexHalf12D, Reduce_half_types);
TYPED_TEST_SUITE(ReduceNoIndexBHalfFloat12D, Reduce_bhalf_float_Types);

TYPED_TEST(ReduceNoIndexFloat, ReduceNoIndexTestFloat_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexFloat, ReduceNoIndexTestFloat_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexFloat, ReduceNoIndexTestFloat_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexDouble, ReduceNoIndexTestDouble_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexDouble, ReduceNoIndexTestDouble_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexDouble, ReduceNoIndexTestDouble_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexInt8, ReduceNoIndexTestInt8_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexInt8, ReduceNoIndexTestInt8_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexInt8, ReduceNoIndexTestInt8_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexHalf, ReduceNoIndexTestHalf_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexHalf, ReduceNoIndexTestHalf_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexHalf, ReduceNoIndexTestHalf_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexBHalfFloat, ReduceNoIndexTesBtHalfFloat_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexBHalfFloat, ReduceNoIndexTestBHalfFloat_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexBHalfFloat, ReduceNoIndexTestBHalfFloat_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexFloat12D, ReduceNoIndexTestFloat_12D_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexFloat12D, ReduceNoIndexTestFloat_12D_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexFloat12D, ReduceNoIndexTestFloat_12D_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexDouble12D, ReduceNoIndexTestDouble_12D_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexDouble12D, ReduceNoIndexTestDouble_12D_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexDouble12D, ReduceNoIndexTestDouble_12D_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexInt812D, ReduceNoIndexTestInt8_12D_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexInt812D, ReduceNoIndexTestInt8_12D_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexInt812D, ReduceNoIndexTestInt8_12D_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexHalf12D, ReduceNoIndexTestHalf_12D_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexHalf12D, ReduceNoIndexTestHalf_12D_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexHalf12D, ReduceNoIndexTestHalf_12D_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}

TYPED_TEST(ReduceNoIndexBHalfFloat12D, ReduceNoIndexTesBtHalfFloat_12D_AMAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::AMAX>();
}

TYPED_TEST(ReduceNoIndexBHalfFloat12D, ReduceNoIndexTestBHalfFloat_12D_MIN)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MIN>();
}

TYPED_TEST(ReduceNoIndexBHalfFloat12D, ReduceNoIndexTestBHalfFloat_12D_MAX)
{
    // trigger Run() -> Generic
    this->template Run<ReduceTensorOp::MAX>();
}
