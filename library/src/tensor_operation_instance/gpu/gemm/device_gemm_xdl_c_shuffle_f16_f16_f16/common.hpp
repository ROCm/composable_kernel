// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType>
class DeviceGemm_Xdl_CShuffle_Appenders final
{
    ~DeviceGemm_Xdl_CShuffle_Appenders() = delete;

    using IDeviceGemm = DeviceGemm<ALayout,
                                   BLayout,
                                   CLayout,
                                   ADataType,
                                   BDataType,
                                   CDataType,
                                   PassThrough,
                                   PassThrough,
                                   PassThrough>;
    using Appender    = std::function<void(std::vector<std::unique_ptr<IDeviceGemm>>&)>;
    using Appenders   = std::vector<Appender>;

    public:
    static Appenders& Get()
    {
        static Appenders appenders;
        return appenders;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
