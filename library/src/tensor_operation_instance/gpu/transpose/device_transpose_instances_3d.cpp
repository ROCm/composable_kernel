// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/transpose/device_transpose_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_transpose_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwise3dImpl<F16, F16, NCDHW, 3>>>& instances)
{
#ifdef CK_ENABLE_FP16
    add_device_operation_instances(instances, device_transpose_f16_instances<F16, F16, NCDHW, 3>{});
#else
    ignore = instances;
#endif
}

void add_device_transpose_f32_instances(
    std::vector<std::unique_ptr<DeviceElementwise3dImpl<F32, F32, NCDHW, 3>>>& instances)
{
#ifdef CK_ENABLE_FP32
    add_device_operation_instances(instances, device_transpose_f32_instances<F32, F32, NCDHW, 3>{});
#else
    ignore = instances;
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
