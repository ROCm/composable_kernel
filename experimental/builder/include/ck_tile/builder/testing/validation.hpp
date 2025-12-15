// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/error.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/utility/type_convert.hpp"
#include <string_view>
#include <vector>
#include <algorithm>
#include <functional>

namespace ck_tile::builder::test {

using ErrorCounter = uint64_t;

struct ValidationReport
{
    struct Case
    {
        std::string tensor_name;
        uint64_t wrong_elements;
        uint64_t total_elements;

        bool is_ok() const { return wrong_elements == 0; }
    };

    template <DataType DT, size_t RANK>
    bool check(std::string_view tensor_name,
               const TensorDescriptor<DT, RANK>& descriptor,
               const void* actual,
               const void* expected,
               double rtol = 1e-3,
               double atol = 1e-3);

    std::vector<Case> get_errors() const
    {
        std::vector<Case> errors;
        std::copy_if(reports_.begin(),
                     reports_.end(),
                     std::back_inserter(errors),
                     [](const auto& report) { return !report.is_ok(); });
        return errors;
    }

    private:
    std::vector<Case> reports_;
};

template <DataType DT, int RANK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) //
    void compare_kernel(const uint64_t numel,
                        const void* actual_data,
                        const void* expected_data,
                        std::array<size_t, RANK> shape_scan,
                        std::array<size_t, RANK> strides,
                        ErrorCounter* error_count,
                        double rtol,
                        double atol)
{
    using CKType = typename factory::internal::DataTypeToCK<DT>::type;

    const auto* actual   = static_cast<const CKType*>(actual_data);
    const auto* expected = static_cast<const CKType*>(expected_data);

    const auto gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for(uint64_t flat_idx = gid; flat_idx < numel; flat_idx += gridDim.x * BLOCK_SIZE)
    {
        size_t offset = 0;
        auto idx      = flat_idx;
        for(int i = 0; i < RANK; ++i)
        {
            const auto scanned_dim = shape_scan[i];
            const auto axis_idx    = idx / scanned_dim;
            idx %= scanned_dim;
            offset += strides[i] * axis_idx;
        }

        static_assert(!std::is_same_v<CKType, double>,
                      "TODO implement compare_kernel() for double");

        const auto o   = static_cast<double>(type_convert<float>(actual[offset]));
        const auto r   = static_cast<double>(type_convert<float>(expected[offset]));
        const auto err = std::abs(o - r);

        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            atomicAdd(error_count, 1);
        }
    }
}

template <DataType DT, int RANK>
bool ValidationReport::check(std::string_view tensor_name,
                             const TensorDescriptor<DT, RANK>& descriptor,
                             const void* actual,
                             const void* expected,
                             double rtol,
                             double atol)
{
    constexpr int block_size = 256;
    const auto kernel        = compare_kernel<DT, RANK, block_size>;

    int occupancy;
    check_hip(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, block_size, 0));

    const auto shape = descriptor.get_lengths();

    std::array<size_t, RANK> shape_scan;
    size_t numel = 1;
    for(int i = RANK; i > 0; --i)
    {
        shape_scan[i - 1] = numel;
        numel *= shape[i - 1];
    }

    std::array<size_t, RANK> strides;
    std::copy(descriptor.get_strides().begin(), descriptor.get_strides().end(), strides.begin());

    // Allocate and reset counter
    auto d_error_count = alloc_buffer(sizeof(ErrorCounter));
    check_hip(hipMemset(d_error_count.get(), 0, sizeof(ErrorCounter)));

    kernel<<<occupancy, block_size>>>(numel,
                                      actual,
                                      expected,
                                      shape_scan,
                                      strides,
                                      reinterpret_cast<ErrorCounter*>(d_error_count.get()),
                                      rtol,
                                      atol);
    check_hip(hipGetLastError());

    ErrorCounter error_count = 0;
    check_hip(
        hipMemcpy(&error_count, d_error_count.get(), sizeof(ErrorCounter), hipMemcpyDeviceToHost));

    reports_.push_back(Case{
        .tensor_name    = std::string(tensor_name),
        .wrong_elements = error_count,
        .total_elements = numel,
    });

    return error_count == 0;
}

} // namespace ck_tile::builder::test
