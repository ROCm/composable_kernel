// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/error.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/utility/type_convert.hpp"
#include <string_view>
#include <vector>

namespace ck_tile::builder::test {

using ErrorCounter = uint64_t;

template <DataType DT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) //
    void flat_compare_kernel(const uint64_t n,
                             const void* actual_data,
                             const void* expected_data,
                             ErrorCounter* error_count,
                             double rtol,
                             double atol)
{
    using CKType = typename factory::internal::DataTypeToCK<DT>::type;

    const auto* actual   = static_cast<const CKType*>(actual_data);
    const auto* expected = static_cast<const CKType*>(expected_data);

    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i = gid; i < n; i += gridDim.x)
    {
        static_assert(!std::is_same_v<CKType, double>,
                      "TODO implement flat_compare_kernel() for double");

        const auto o   = static_cast<double>(type_convert<float>(actual[i]));
        const auto r   = static_cast<double>(type_convert<float>(expected[i]));
        const auto err = std::abs(o - r);

        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            atomicAdd(error_count, 1);
        }
    }
}

template <DataType DT>
bool compare_tensors(std::string_view tensor_name,
                     const TensorDescriptor<DT>& descriptor,
                     const void* actual,
                     const void* expected,
                     double rtol = 1e-3,
                     double atol = 1e-3)
{
    constexpr int block_size = 256;
    const auto kernel        = flat_compare_kernel<DT, block_size>;

    int occupancy;
    check_hip(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, block_size, 0));

    const size_t num_elements = descriptor.get_element_space_size();

    auto error_count = alloc_buffer(sizeof(ErrorCounter));

    kernel<<<occupancy, block_size>>>(
        num_elements, actual, expected, reinterpret_cast<uint64_t*>(error_count.get()), rtol, atol);
    check_hip(hipGetLastError());

    ErrorCounter h_error_count = 0;
    check_hip(
        hipMemcpy(&h_error_count, error_count.get(), sizeof(ErrorCounter), hipMemcpyDeviceToHost));

    if(h_error_count != 0)
    {
        std::cerr << "tensor " << tensor_name << " does not match reference [" << h_error_count
                  << "/" << num_elements << " errors]" << std::endl;
    }

    return h_error_count == 0;
}

} // namespace ck_tile::builder::test
