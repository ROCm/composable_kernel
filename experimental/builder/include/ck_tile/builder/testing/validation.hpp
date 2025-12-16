// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/error.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/testing/tensor_foreach.hpp"
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

template <DataType DT, size_t RANK>
bool ValidationReport::check(std::string_view tensor_name,
                             const TensorDescriptor<DT, RANK>& descriptor,
                             const void* actual_data,
                             const void* expected_data,
                             double rtol,
                             double atol)
{
    const auto strides = descriptor.get_strides();

    // During development and CI, only the kernels that were changed would fail, and so we can
    // assume that the average case does not have errors. Therefore, split out testing into a
    // quick test which just counts the incorrect elements, and a more in-depth test that also
    // returns the indices of the incorrect items.

    // Initial pass: count errors

    // Allocate and reset counter
    auto d_error_count = alloc_buffer(sizeof(ErrorCounter));
    check_hip(hipMemset(d_error_count.get(), 0, sizeof(ErrorCounter)));

    tensor_foreach(descriptor.get_lengths(), [=, error_count = d_error_count.get()](auto index) {
        using CKType = typename factory::internal::DataTypeToCK<DT>::type;

        const auto* actual   = static_cast<const CKType*>(actual_data);
        const auto* expected = static_cast<const CKType*>(expected_data);

        static_assert(!std::is_same_v<CKType, double>,
                      "TODO implement compare_kernel() for double");

        const auto offset = calculate_offset(index, strides);

        const auto o   = static_cast<double>(type_convert<float>(actual[offset]));
        const auto r   = static_cast<double>(type_convert<float>(expected[offset]));
        const auto err = std::abs(o - r);

        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            // We expect the number of errors to be very low, so just use an atomic
            // for now.
            atomicAdd(reinterpret_cast<ErrorCounter*>(error_count), 1);
        }
    });

    ErrorCounter error_count = 0;
    check_hip(
        hipMemcpy(&error_count, d_error_count.get(), sizeof(ErrorCounter), hipMemcpyDeviceToHost));

    // TODO: Gather detailed coordinates.

    reports_.push_back(Case{
        .tensor_name    = std::string(tensor_name),
        .wrong_elements = error_count,
        .total_elements = descriptor.get_element_size(),
    });

    return error_count == 0;
}

} // namespace ck_tile::builder::test
