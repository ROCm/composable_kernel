/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_HOST_REDUCE_UTIL_HPP
#define GUARD_HOST_REDUCE_UTIL_HPP

#include <half.hpp>
#include <limits>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>

#include "reduction_enums.hpp"

namespace ck {

namespace host_reduce {

using ck::NanPropagation_t;
using ck::ReduceTensorOp_t;

template <typename T>
static inline bool float_equal_one(T);

static inline bool float_equal_one(float x) { return x == 1.0f; };

static inline bool float_equal_one(double x) { return x == 1.0; };

static inline bool float_equal_one(half_float::half x)
{
    return x == static_cast<half_float::half>(1.0f);
};

template <typename T>
static inline bool float_equal_zero(T x);

static inline bool float_equal_zero(float x) { return x == 0.0f; };

static inline bool float_equal_zero(double x) { return x == 0.0; };

static inline bool float_equal_zero(half_float::half x)
{
    return x == static_cast<half_float::half>(0.0f);
};

template <typename compType, ReduceTensorOp_t ReduceOpId>
__host__ static inline std::function<void(compType&)> PreUnaryOpFn(int)
{
    using std::abs;

    if constexpr(ReduceOpId == ReduceTensorOp_t::NORM1)
    {
        return ([&](compType& a_) { a_ = abs(a_); });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::NORM2)
    {
        return ([&](compType& a_) { a_ = a_ * a_; });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::AMAX)
    {
        return ([&](compType& a_) { a_ = abs(a_); });
    }
    else
    {
        // ReduceTensorOp_t::AVG:
        // ReduceTensorOp_t::ADD:
        // ReduceTensorOp_t::MUL:
        // ReduceTensorOp_t::MIN:
        // ReduceTensorOp_t::MAX:
        return ([&](compType&) {});
    };
};

template <typename compType, ReduceTensorOp_t ReduceOpId>
__host__ static inline std::function<void(compType&)> PosUnaryOpFn(int divider)
{
    using std::sqrt;

    if constexpr(ReduceOpId == ReduceTensorOp_t::NORM2)
    {
        return ([&](compType& a_) { a_ = sqrt(a_); });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::AVG)
    {
        return ([&, divider](compType& a_) {
            a_ = a_ / static_cast<compType>(static_cast<float>(divider));
        });
    }
    else
    {
        // ReduceTensorOp_t::ADD:
        // ReduceTensorOp_t::NORM1:
        // ReduceTensorOp_t::MUL:
        // ReduceTensorOp_t::MIN:
        // ReduceTensorOp_t::MAX:
        // ReduceTensorOp_t::AMAX:
        return ([&](compType&) {});
    }
};

template <typename compType, ReduceTensorOp_t ReduceOpId>
__host__ static inline std::function<void(compType&, compType)> ReduceOpFn()
{
    if constexpr(ReduceOpId == ReduceTensorOp_t::ADD || ReduceOpId == ReduceTensorOp_t::AVG ||
                 ReduceOpId == ReduceTensorOp_t::NORM1 || ReduceOpId == ReduceTensorOp_t::NORM2)
    {
        return ([&](compType& a_, compType b_) { a_ = a_ + b_; });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::MUL)
    {
        return ([&](compType& a_, compType b_) { a_ = a_ * b_; });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::MIN)
    {
        return ([&](compType& a_, compType b_) {
            if(a_ > b_)
                a_ = b_;
        });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::MAX || ReduceOpId == ReduceTensorOp_t::AMAX)
    {
        return ([&](compType& a_, compType b_) {
            if(a_ < b_)
                a_ = b_;
        });
    }
};

template <typename compType, ReduceTensorOp_t ReduceOpId>
__host__ static inline std::function<void(compType&, compType, bool& changed)> ReduceOpFn2()
{
    if constexpr(ReduceOpId == ReduceTensorOp_t::MIN)
    {
        return ([&](compType& a_, compType b_, bool& changed) {
            if(a_ > b_)
            {
                a_      = b_;
                changed = true;
            }
            else
                changed = false;
        });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::MAX || ReduceOpId == ReduceTensorOp_t::AMAX)
    {
        return ([&](compType& a_, compType b_, bool& changed) {
            if(a_ < b_)
            {
                a_      = b_;
                changed = true;
            }
            else
                changed = false;
        });
    }
    else
    {
        // ReduceTensorOp_t::ADD:
        // ReduceTensorOp_t::MUL:
        // ReduceTensorOp_t::AVG:
        // ReduceTensorOp_t::NORM1:
        // ReduceTensorOp_t::NORM2:
        return (std::function<void(compType&, compType, bool&)>{});
    };
};

template <typename compType, ReduceTensorOp_t ReduceOpId>
__host__ static inline compType ReduceOpZeroVal()
{
    if constexpr(ReduceOpId == ReduceTensorOp_t::MUL)
    {
        return (static_cast<compType>(1.0f));
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::MIN)
    {
        return (std::numeric_limits<compType>::max());
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::MAX)
    {
        return (std::numeric_limits<compType>::lowest());
    }
    else if constexpr(ReduceOpId == ReduceTensorOp_t::AMAX)
    {
        return (static_cast<compType>(0.0f));
    }
    else
    {
        // ReduceTensorOp_t::ADD
        // ReduceTensorOp_t::AVG
        // ReduceTensorOp_t::NORM1
        // ReduceTensorOp_t::NORM2
        return (static_cast<compType>(0.0f));
    };
};

template <typename compType, bool PropagateNan>
__host__ static inline void binop_with_nan_check(std::function<void(compType&, compType)> opReduce,
                                                 compType& accuVal,
                                                 compType currVal)
{
    using std::isnan;

    if constexpr(!PropagateNan)
    {
        opReduce(accuVal, currVal);
    }
    else
    {
        if(isnan(currVal))
            accuVal = currVal;
        else
            opReduce(accuVal, currVal);
    };
};

template <typename compType, bool PropagateNan>
__host__ static inline void
binop_with_nan_check2(std::function<void(compType&, compType, bool&)> opReduce,
                      compType& accuVal,
                      compType currVal,
                      int& accuIndex,
                      int currIndex)
{
    using std::isnan;

    if constexpr(!PropagateNan)
    {
        bool changed;

        opReduce(accuVal, currVal, changed);

        if(changed)
            accuIndex = currIndex;
    }
    else
    {
        if(isnan(currVal))
        {
            accuVal   = currVal;
            accuIndex = currIndex;
        }
        else
        {
            bool changed;

            opReduce(accuVal, currVal, changed);

            if(changed)
                accuIndex = currIndex;
        };
    };
};

}; // namespace host_reduce

static inline std::vector<int> to_int_vector(const std::vector<size_t>& inData)
{
    std::vector<int> outData;

    for(auto elem : inData)
        outData.push_back(static_cast<int>(elem));

    return (outData);
};

}; // namespace ck

#endif
