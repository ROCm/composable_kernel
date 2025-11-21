// Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <numeric>
#include <hip/hip_runtime.h>
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/testing/conv_args.hpp"
#include "ck_tile/builder/testing/type_traits.hpp"
#include "ck_tile/builder/conv_factory.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

namespace ck_tile::builder::test {

struct DeviceMemoryDeleter
{
    void operator()(std::byte* ptr) const
    {
        if(ptr)
            (void)hipFree(ptr);
    }
};

using DeviceBuffer = std::unique_ptr<std::byte[], DeviceMemoryDeleter>;

template <DataType DT>
DeviceBuffer alloc_tensor(ck::HostTensorDescriptor descriptor)
{
    const auto total_elements = descriptor.GetElementSpaceSize();
    const auto total_size     = total_elements * sizeof(typename DataTypeTraits<DT>::Type);

    std::byte* d_buf  = nullptr;
    const auto status = hipMalloc(&d_buf, total_size);
    // TODO(Robin): How to check error without relying on google test?
    // Ideally we get some sort of trace here, but thats not possible until c++23.
    // For now just throw a runtime error.
    if(status != hipSuccess)
    {
        throw std::runtime_error("failed to load hip memory");
    }
    return DeviceBuffer(d_buf);
}

template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE>
struct TensorMemoryManager
{
    // Type aliases for tensor data types
    // For now, all tensors use the same data type from the signature
    using InputDataType  = DataTypeTraits<SIGNATURE.data_type>::Type;
    using WeightDataType = DataTypeTraits<SIGNATURE.data_type>::Type;
    using OutputDataType = DataTypeTraits<SIGNATURE.data_type>::Type;

    using Layouts =
        decltype(ck_tile::builder::factory_internal::GetTensorLayout<SIGNATURE.layout,
                                                                     SIGNATURE.spatial_dim,
                                                                     ConvDirection::FORWARD>());

    TensorMemoryManager(const ConvArgs<SIGNATURE>& args)
        : param(args.to_conv_param()),
          input_descriptor(ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
                           typename Layouts::ALayout>(this->param)),
          weight_descriptor(ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
                            typename Layouts::BLayout>(this->param)),
          output_descriptor(ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
                            typename Layouts::ELayout>(this->param)),
          input_buf(alloc_tensor<SIGNATURE.data_type>(this->input_descriptor)),
          weight_buf(alloc_tensor<SIGNATURE.data_type>(this->weight_descriptor)),
          output_buf(alloc_tensor<SIGNATURE.data_type>(this->output_descriptor))
    {
    }

    ck::utils::conv::ConvParam param;

    ck::HostTensorDescriptor input_descriptor;
    ck::HostTensorDescriptor weight_descriptor;
    ck::HostTensorDescriptor output_descriptor;

    // Device memory buffers
    DeviceBuffer input_buf  = nullptr;
    DeviceBuffer weight_buf = nullptr;
    DeviceBuffer output_buf = nullptr;
};

} // namespace ck_tile::builder::test
