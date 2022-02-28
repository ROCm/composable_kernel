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

#include "reduction_common.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace detail {

template <typename T>
static inline __device__ bool is_nan(T x)
{
    return (isnan(x));
};

template <>
inline __device__ bool is_nan<half_t>(half_t x)
{
    return (__hisnan(x));
};

template <bool propagate_nan, typename opReduce, typename AccDataType>
struct accumulate_with_nan_check;

template <typename opReduce, typename AccDataType>
struct accumulate_with_nan_check<false, opReduce, AccDataType>
{
    // cppcheck-suppress constParameter
    __device__ static inline void calculate(AccDataType& accuVal, AccDataType currVal)
    {
        opReduce{}(accuVal, currVal);
    };
};

template <typename opReduce, typename AccDataType>
struct accumulate_with_nan_check<true, opReduce, AccDataType>
{
    __device__ static inline void calculate(AccDataType& accuVal, AccDataType currVal)
    {
        if(is_nan(currVal))
            accuVal = currVal;
        else
            opReduce{}(accuVal, currVal);
    };
};

template <bool propagate_nan, typename opReduce, typename AccDataType, typename IndexDataType>
struct accumulate_with_indices_with_nan_check;

template <typename opReduce, typename AccDataType, typename IndexDataType>
struct accumulate_with_indices_with_nan_check<false, opReduce, AccDataType, IndexDataType>
{
    __device__ static inline void
    // cppcheck-suppress constParameter
    calculate(AccDataType& accuVal,
              AccDataType currVal,
              IndexDataType& accuIndex,
              IndexDataType currIndex)
    {
        bool changed = false;

        opReduce{}(accuVal, currVal, changed);

        if(changed)
            accuIndex = currIndex;
    };
};

template <typename opReduce, typename AccDataType, typename IndexDataType>
struct accumulate_with_indices_with_nan_check<true, opReduce, AccDataType, IndexDataType>
{
    // The method is called when the opReduce is indexable and the user asked for indices
    __device__ static inline void calculate(AccDataType& accuVal,
                                            AccDataType currVal,
                                            IndexDataType& accuIndex,
                                            IndexDataType currIndex)
    {
        if(is_nan(currVal))
        {
            accuVal   = currVal;
            accuIndex = currIndex;
        }
        else
        {
            bool changed = false;

            opReduce{}(accuVal, currVal, changed);

            if(changed)
                accuIndex = currIndex;
        }
    };
};

}; // namespace detail
}; // end of namespace ck

#endif
