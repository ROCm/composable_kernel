// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include "ck/ck.hpp"
#include "ck/utility/type.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace utils {

template <typename Layout>
inline void
validate_gemm_stride(int M, int N, int stride, const std::string& stride_name = "Stride")
{
    if(ck::is_same_v<Layout, ck::tensor_layout::gemm::ColumnMajor>)
    {
        if(stride < M)
        {
            throw std::runtime_error(
                "Error: For ColumnMajor layout, " + stride_name + " (" + std::to_string(stride) +
                ") must be greater than or equal to dim (" + std::to_string(M) + ")");
        }
    }
    else // RowMajor
    {
        if(stride < N)
        {
            throw std::runtime_error(
                "Error: For RowMajor layout, " + stride_name + " (" + std::to_string(stride) +
                ") must be greater than or equal to dim (" + std::to_string(N) + ")");
        }
    }
}

// Convenience functions for common GEMM patterns
template <typename ALayout, typename BLayout, typename CLayout>
inline void validate_gemm_strides_abc(int M, int N, int K, int StrideA, int StrideB, int StrideC)
{
    validate_gemm_stride<ALayout>(M, K, StrideA, "StrideA");
    validate_gemm_stride<BLayout>(K, N, StrideB, "StrideB");
    validate_gemm_stride<CLayout>(M, N, StrideC, "StrideC");
}

} // namespace utils
} // namespace ck
