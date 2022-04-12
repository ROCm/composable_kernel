/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef CK_GRIDWISE_2D_COMPUTE_MEAN_AND_MEANSQUARE_USING_REDUCE_BLOCKWISE_HPP
#define CK_GRIDWISE_2D_COMPUTE_MEAN_AND_MEANSQUARE_USING_REDUCE_BLOCKWISE_HPP

#include "data_type.hpp"
#include "gridwise_2d_reduction_blockwise.hpp"

namespace ck {

template <typename GridwiseReduceMean,
          typename GridwiseReduceMeanSquare,
          typename InOutDataType,
          typename AccDataType,
          typename InOutGridDesc_M_K,
          typename ScaleBiasMeanVarGridDesc_M,
          typename InElementwiseOperationMean,
          typename AccElementwiseOperationMean,
          typename InElementwiseOperationMeanSquare,
          typename AccElementwiseOperationMeanSquare>
__global__ void kernel_compute_mean_and_meansquare_using_reduction_blockwise(
    const InOutGridDesc_M_K in_out_grid_desc_m_k,
    const ScaleBiasMeanVarGridDesc_M scale_bias_mean_var_grid_desc_m,
    const InElementwiseOperationMean in_element_wise_op_mean,
    const AccElementwiseOperationMean acc_element_wise_op_mean,
    const InElementwiseOperationMeanSquare in_element_wise_op_meansquare,
    const AccElementwiseOperationMeanSquare acc_element_wise_op_meansquare,
    const InOutDataType* const __restrict__ p_in_global,
    AccDataType* const __restrict__ resultSaveMean,
    AccDataType* const __restrict__ resultSaveMeanSquare)
{
    constexpr bool IsSecondCall = false;

    GridwiseReduceMean::template Run<IsSecondCall>(in_out_grid_desc_m_k,
                                                   scale_bias_mean_var_grid_desc_m,
                                                   in_element_wise_op_mean,
                                                   acc_element_wise_op_mean,
                                                   type_convert<AccDataType>(1.0f),
                                                   p_in_global,
                                                   type_convert<AccDataType>(0.0f),
                                                   resultSaveMean,
                                                   nullptr,
                                                   nullptr);

    GridwiseReduceMeanSquare::template Run<IsSecondCall>(in_out_grid_desc_m_k,
                                                         scale_bias_mean_var_grid_desc_m,
                                                         in_element_wise_op_meansquare,
                                                         acc_element_wise_op_meansquare,
                                                         type_convert<AccDataType>(1.0f),
                                                         p_in_global,
                                                         type_convert<AccDataType>(0.0f),
                                                         resultSaveMeanSquare,
                                                         nullptr,
                                                         nullptr);
};

} // namespace ck
#endif
