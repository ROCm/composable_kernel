// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/device_avgpool_bwd.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DOutDataType,
          typename DInDataType,
          typename ComputeDataType,
          ck::index_t BlockSize,
          ck::index_t MThreadClusterSize,
          ck::index_t KThreadClusterSize,
          ck::index_t MThreadSliceSize,
          ck::index_t KThreadSliceSize,
          ck::index_t InSrcOutDstVectorSize>
struct DeviceAvgPool3dBwdImpl : public DeviceAvgPoolBwd<DOutDataType, DInDataType>
{
    struct Argument : public BaseArgument
    {
        Argument() {}
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            ignore = p_arg;
            ignore = stream_config;
            return 0;
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        ignore = arg;
        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_dout,
                        void* p_din,
                        std::vector<ck::index_t> dout_n_k_wos_lengths,
                        std::vector<ck::index_t> din_n_k_wos_length,
                        std::vector<ck::index_t> window_k_c_xs_lengths,
                        std::vector<ck::index_t> dout_n_k_wos_strides,
                        std::vector<ck::index_t> din_n_k_wos_strides,
                        std::vector<ck::index_t> window_strides,
                        std::vector<ck::index_t> window_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads) override
    {
        ignore = p_dout;
        ignore = p_din;
        ignore = dout_n_k_wos_lengths;
        ignore = dout_n_k_wos_strides;
        ignore = din_n_k_wos_length;
        ignore = din_n_k_wos_strides;
        ignore = window_k_c_xs_lengths;
        ignore = window_strides;
        ignore = window_dilations;
        ignore = input_left_pads;
        ignore = input_right_pads;

        return std::make_unique<Argument>();
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceAvgPool3dBwd<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str <<"InSrcOutDstVectorSize_" << InSrcOutDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
