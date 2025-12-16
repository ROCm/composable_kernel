// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <memory>
#include <numeric>
#include <span>
#include <concepts>
#include <hip/hip_runtime.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/testing/type_traits.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/device_tensor_generator.hpp"

namespace ck_tile::builder::test {

template <DataType DT>
void init_tensor_buffer_uniform_int(const DeviceBuffer& buf,
                                    const TensorDescriptor<DT>& descriptor,
                                    int min_val,
                                    int max_val)
{
    size_t size = descriptor.get_element_space_size_in_bytes();

    if(max_val - min_val <= 1)
    {
        throw std::runtime_error("Error while filling device tensor with random integer data: max "
                                 "value must be at least 2 greater than min value, otherwise "
                                 "tensor will be filled by a constant value (end is exclusive)");
    }

    using ck_type = factory::internal::DataTypeToCK<DT>::type;

    // we might be asked to generate int values on fp data types that don't have the required
    // precision
    if(static_cast<ck_type>(max_val - 1) == static_cast<ck_type>(min_val))
    {
        throw std::runtime_error("Error while filling device tensor with random integer data: "
                                 "insufficient precision in specified range");
    }
    size_t packed_size = ck::packed_size_v<ck_type>;
    fill_tensor_uniform_rand_int_values<<<256, 256>>>(
        static_cast<ck_type>(buf.get()), min_val, max_val, (size * packed_size) / sizeof(ck_type));
}

template <DataType DT>
void init_tensor_buffer_uniform_fp(const DeviceBuffer& buf,
                                   const TensorDescriptor<DT>& descriptor,
                                   float min_value,
                                   float max_value)
{
    size_t size = descriptor.get_element_space_size_in_bytes();

    using ck_type = factory::internal::DataTypeToCK<DT>::type;

    size_t packed_size = ck::packed_size_v<ck_type>;
    fill_tensor_uniform_rand_fp_values<<<256, 256>>>(reinterpret_cast<ck_type*>(buf.get()),
                                                     min_value,
                                                     max_value,
                                                     (size * packed_size) / sizeof(ck_type));
}

template <DataType DT>
void init_tensor_buffer_normal_fp(const DeviceBuffer& buf,
                                  const TensorDescriptor<DT>& descriptor,
                                  float sigma,
                                  float mean)
{
    size_t size = descriptor.get_element_space_size_in_bytes();

    using ck_type      = factory::internal::DataTypeToCK<DT>::type;
    size_t packed_size = ck::packed_size_v<ck_type>;
    fill_tensor_norm_rand_fp_values<<<256, 256>>>(
        static_cast<ck_type*>(buf.get()), sigma, mean, (size * packed_size) / sizeof(ck_type));
}

} // namespace ck_tile::builder::test
