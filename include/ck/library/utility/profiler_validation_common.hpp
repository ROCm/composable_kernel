#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include "ck/ck.hpp"
#include "ck/utility/type.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace profiler {

template <typename Layout>
inline void
validate_matrix_stride(int M, int N, int stride, const std::string& matrix_name = "Stride")
{
    if(ck::is_same_v<Layout, ck::tensor_layout::gemm::ColumnMajor>)
    {
        if(stride < M)
        {
            throw std::runtime_error(
                "Error: For ColumnMajor layout, " + matrix_name + " (" + std::to_string(stride) +
                ") must be greater than or equal to dim (" + std::to_string(M) + ")");
        }
    }
    else // RowMajor
    {
        if(stride < N)
        {
            throw std::runtime_error(
                "Error: For RowMajor layout, " + matrix_name + " (" + std::to_string(stride) +
                ") must be greater than or equal to dim (" + std::to_string(N) + ")");
        }
    }
}

// Convenience functions for common GEMM patterns
template <typename ALayout, typename BLayout, typename CLayout>
inline void validate_gemm_strides(int M, int N, int K, int StrideA, int StrideB, int StrideC)
{
    validate_matrix_stride<ALayout>(M, K, StrideA, "StrideA");
    validate_matrix_stride<BLayout>(K, N, StrideB, "StrideB");
    validate_matrix_stride<CLayout>(M, N, StrideC, "StrideC");
}

template <typename Layout>
inline void validate_batch_stride(
    int dim1, int dim2, int stride, int batch_stride, const std::string& matrix_name = "Matrix")
{
    // validate regular stride
    validate_matrix_stride<Layout>(dim1, dim2, stride, matrix_name);

    // validate batch stride
    int min_batch_stride = dim1 * stride;
    if(batch_stride < min_batch_stride)
    {
        throw std::runtime_error("Error: Batch" + matrix_name + " (" +
                                 std::to_string(batch_stride) +
                                 ") must be >= " + std::to_string(min_batch_stride));
    }
}

} // namespace profiler
} // namespace ck
