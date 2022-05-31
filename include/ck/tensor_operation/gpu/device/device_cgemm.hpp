/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#pragma once
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceCGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a_real,
                                                              const void* p_a_imag,
                                                              const void* p_b_real,
                                                              const void* p_b_imag,
                                                              void* p_c_real,
                                                              void* p_c_imag,
                                                              void* p_workspace,
                                                              ck::index_t M,
                                                              ck::index_t N,
                                                              ck::index_t K,
                                                              ck::index_t StrideA,
                                                              ck::index_t StrideB,
                                                              ck::index_t StrideC,
                                                              AElementwiseOperation a_element_op,
                                                              BElementwiseOperation b_element_op,
                                                              CElementwiseOperation c_element_op,
                                                              ck::index_t KBatch = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
    virtual std::size_t GetWorkspaceSize(index_t MRaw,
                                         index_t NRaw,
                                         index_t KRaw,
                                         index_t StrideA,
                                         index_t StrideB,
                                         index_t StrideC)     = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceCGemmPtr = std::unique_ptr<
    DeviceCGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
