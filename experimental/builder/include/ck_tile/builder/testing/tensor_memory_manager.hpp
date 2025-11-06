// Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <hip/hip_runtime.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::testing {

struct DeviceMemoryDeleter
{
    void operator()(std::byte* ptr) const
    {
        if(ptr)
            (void)hipFree(ptr);
    }
};

using DeviceBuffer = std::unique_ptr<std::byte[], DeviceMemoryDeleter>;

template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE>
struct TensorMemoryManager
{
    // Type aliases for tensor data types
    // For now, all tensors use the same data type from the signature
    using InputDataType  = decltype(SIGNATURE.data_type);
    using WeightDataType = decltype(SIGNATURE.data_type);
    using OutputDataType = decltype(SIGNATURE.data_type);

    TensorMemoryManager() = default;

    // Device memory buffers
    DeviceBuffer input_buf  = nullptr;
    DeviceBuffer weight_buf = nullptr;
    DeviceBuffer output_buf = nullptr;
};

} // namespace ck_tile::builder::testing
