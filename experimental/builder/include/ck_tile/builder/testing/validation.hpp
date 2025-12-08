// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <string_view>
#include <sstream>
#include <vector>

namespace ck_tile::builder::test {

template <DataType DT>
bool compare_tensors(std::string_view tensor_name,
                     const TensorDescriptor<DT>& descriptor,
                     const void* actual,
                     const void* expected,
                     double rtol = 1e-3,
                     double atol = 1e-3)
{
    using CKType = typename factory::internal::DataTypeToCK<DT>::type;

    const size_t num_elements = descriptor.get_element_space_size();

    std::vector<CKType> h_actual(num_elements);
    std::vector<CKType> h_expected(num_elements);

    HIP_CHECK_ERROR(hipMemcpy(h_actual.data(), actual, h_actual.size(), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(
        hipMemcpy(h_expected.data(), expected, h_expected.size(), hipMemcpyDeviceToHost));

    std::stringstream msg;
    msg << "error: " << tensor_name << " does not match reference";
    return ck::utils::check_err(h_expected, h_actual, msg.str(), rtol, atol);
}

} // namespace ck_tile::builder::test
