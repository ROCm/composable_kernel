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
#ifndef CK_REDUCTION_FUNCTIONS_BINOP_HPP
#define CK_REDUCTION_FUNCTIONS_BINOP_HPP

#include "data_type.hpp"
#include "math_v2.hpp"

#include "reduction_common.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace detail {

// Check for NaN; guarantee NaNs are NOT propagated to result (i.e., ignore NaNs)
template <typename ReduceOperation, typename AccDataType>
struct AccumulateWithNanIgnore
{
    __device__ static inline void Calculate(AccDataType& accuVal, AccDataType currVal)
    {
        if(!isnan(currVal))
        {
            ReduceOperation{}(accuVal, currVal);
        }
    };
};

template <bool PropagateNan, typename ReduceOperation, typename AccDataType>
struct AccumulateWithNanCheck;

// Does not check for NaN; does not guarantee NaNs be propagated to result
// e.g., given that max(a, b) = a > b ? a : b
// then  max(NaN, 1) returns 1
//       max(1, NaN) returns NaN
// since any comparison involving NaNs returns false
template <typename ReduceOperation, typename AccDataType>
struct AccumulateWithNanCheck<false, ReduceOperation, AccDataType>
{
    // cppcheck-suppress constParameter
    __host__ __device__ static inline void Calculate(AccDataType& accuVal, AccDataType currVal)
    {
        ReduceOperation{}(accuVal, currVal);
    };
};

// Check for NaN; guarantees NaNs be propagated to result
template <typename ReduceOperation, typename AccDataType>
struct AccumulateWithNanCheck<true, ReduceOperation, AccDataType>
{
    __host__ __device__ static inline void Calculate(AccDataType& accuVal, AccDataType currVal)
    {
        using ck::math::isnan;

        if(isnan(currVal))
        {
            accuVal = currVal;
        }
        else
        {
            ReduceOperation{}(accuVal, currVal);
        };
    };
};

template <bool PropagateNan, typename ReduceOperation, typename AccDataType, typename IndexDataType>
struct AccumulateWithIndexAndNanCheck;

template <typename ReduceOperation, typename AccDataType, typename IndexDataType>
struct AccumulateWithIndexAndNanCheck<false, ReduceOperation, AccDataType, IndexDataType>
{
    __host__ __device__ static inline void
    // cppcheck-suppress constParameter
    Calculate(AccDataType& accuVal,
              AccDataType currVal,
              IndexDataType& accuIndex,
              IndexDataType currIndex)
    {
        bool changed = false;

        ReduceOperation{}(accuVal, currVal, changed);

        if(changed)
            accuIndex = currIndex;
    };
};

template <typename ReduceOperation, typename AccDataType, typename IndexDataType>
struct AccumulateWithIndexAndNanCheck<true, ReduceOperation, AccDataType, IndexDataType>
{
    // The method is called when the ReduceOperation is indexable and the user asked for indices
    __host__ __device__ static inline void Calculate(AccDataType& accuVal,
                                                     AccDataType currVal,
                                                     IndexDataType& accuIndex,
                                                     IndexDataType currIndex)
    {
        using ck::math::isnan;

        if(isnan(currVal))
        {
            accuVal   = currVal;
            accuIndex = currIndex;
        }
        else
        {
            bool changed = false;

            ReduceOperation{}(accuVal, currVal, changed);

            if(changed)
                accuIndex = currIndex;
        }
    };
};

}; // namespace detail
}; // end of namespace ck

#endif
