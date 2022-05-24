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

#include <limits>
#include <cmath>
#include <functional>

#include "reduction_enums.hpp"
#include "data_type.hpp"
#include "math_v2.hpp"

namespace ck {

namespace host_reduce {

using ck::NanPropagation;
using ck::ReduceTensorOp;

template <typename AccDataType, ReduceTensorOp ReduceOpId>
__host__ static inline std::function<void(AccDataType&)> PreUnaryOpFn(int)
{
    using ck::math::abs;

    if constexpr(ReduceOpId == ReduceTensorOp::NORM1)
    {
        return ([&](AccDataType& a_) { a_ = abs(a_); });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::NORM2)
    {
        return ([&](AccDataType& a_) { a_ = a_ * a_; });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::AMAX)
    {
        return ([&](AccDataType& a_) { a_ = abs(a_); });
    }
    else
    {
        // ReduceTensorOp::AVG:
        // ReduceTensorOp::ADD:
        // ReduceTensorOp::MUL:
        // ReduceTensorOp::MIN:
        // ReduceTensorOp::MAX:
        return ([&](AccDataType&) {});
    };
};

template <typename AccDataType, ReduceTensorOp ReduceOpId>
__host__ static inline std::function<void(AccDataType&)> PosUnaryOpFn(int32_t divider)
{
    using std::sqrt;

    if constexpr(ReduceOpId == ReduceTensorOp::NORM2)
    {
        return ([&](AccDataType& a_) { a_ = sqrt(a_); });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::AVG)
    {
        return ([&, divider](AccDataType& a_) {
            a_ = a_ / static_cast<AccDataType>(static_cast<float>(divider));
        });
    }
    else
    {
        // ReduceTensorOp::ADD:
        // ReduceTensorOp::NORM1:
        // ReduceTensorOp::MUL:
        // ReduceTensorOp::MIN:
        // ReduceTensorOp::MAX:
        // ReduceTensorOp::AMAX:
        return ([&](AccDataType&) {});
    }
};

template <typename AccDataType, ReduceTensorOp ReduceOpId>
__host__ static inline std::function<void(AccDataType&, AccDataType)> ReduceOpFn()
{
    if constexpr(ReduceOpId == ReduceTensorOp::ADD || ReduceOpId == ReduceTensorOp::AVG ||
                 ReduceOpId == ReduceTensorOp::NORM1 || ReduceOpId == ReduceTensorOp::NORM2)
    {
        return ([&](AccDataType& a_, AccDataType b_) { a_ = a_ + b_; });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::MUL)
    {
        return ([&](AccDataType& a_, AccDataType b_) { a_ = a_ * b_; });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::MIN)
    {
        return ([&](AccDataType& a_, AccDataType b_) {
            if(a_ > b_)
                a_ = b_;
        });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::MAX || ReduceOpId == ReduceTensorOp::AMAX)
    {
        return ([&](AccDataType& a_, AccDataType b_) {
            if(a_ < b_)
                a_ = b_;
        });
    }
};

template <typename AccDataType, ReduceTensorOp ReduceOpId>
__host__ static inline std::function<void(AccDataType&, AccDataType, bool& changed)> ReduceOpFn2()
{
    if constexpr(ReduceOpId == ReduceTensorOp::MIN)
    {
        return ([&](AccDataType& a_, AccDataType b_, bool& changed) {
            if(a_ > b_)
            {
                a_      = b_;
                changed = true;
            }
            else
                changed = false;
        });
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::MAX || ReduceOpId == ReduceTensorOp::AMAX)
    {
        return ([&](AccDataType& a_, AccDataType b_, bool& changed) {
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
        // ReduceTensorOp::ADD:
        // ReduceTensorOp::MUL:
        // ReduceTensorOp::AVG:
        // ReduceTensorOp::NORM1:
        // ReduceTensorOp::NORM2:
        return (std::function<void(AccDataType&, AccDataType, bool&)>{});
    };
};

template <typename AccDataType, ReduceTensorOp ReduceOpId>
__host__ static inline AccDataType ReduceOpZeroVal()
{
    if constexpr(ReduceOpId == ReduceTensorOp::MUL)
    {
        return (static_cast<AccDataType>(1.0f));
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::MIN)
    {
        return (ck::NumericLimits<AccDataType>::Max());
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::MAX)
    {
        return (ck::NumericLimits<AccDataType>::Lowest());
    }
    else if constexpr(ReduceOpId == ReduceTensorOp::AMAX)
    {
        return (static_cast<AccDataType>(0.0f));
    }
    else
    {
        // ReduceTensorOp::ADD
        // ReduceTensorOp::AVG
        // ReduceTensorOp::NORM1
        // ReduceTensorOp::NORM2
        return (static_cast<AccDataType>(0.0f));
    };
};

template <typename AccDataType, bool PropagateNan>
__host__ static inline void
binop_with_nan_check(std::function<void(AccDataType&, AccDataType)> opReduce,
                     AccDataType& accuVal,
                     AccDataType currVal)
{
    using ck::math::isnan;

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

template <typename AccDataType, typename IndexDataType, bool PropagateNan>
__host__ static inline void
binop_with_index_and_nan_check(std::function<void(AccDataType&, AccDataType, bool&)> opReduce,
                               AccDataType& accuVal,
                               AccDataType currVal,
                               IndexDataType& accuIndex,
                               IndexDataType currIndex)
{
    using ck::math::isnan;

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

}; // namespace ck

#endif
