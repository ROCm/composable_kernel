// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_layernorm_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Pass = ck::tensor_operation::element_wise::PassThrough;

template <typename OutElementwise, index_t Rank, index_t Reduce>
using device_layernorm_f16_instances = std::tuple<
    // clang-format off
        // XDataType, GammaDataType, BetaDataType, AccDataType, YDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorDim, GammaSrcVectorSize, BetaSrcVectorDim, BetaSrcVectorSize, YDstVectorSize>
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 8, 32, 1, 8, 1, 1, 1, 1, 1, 1, 1>, // fallback kernel
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 8, 32, 1, 8, 1, 2, 1, 2, 1, 2, 2>, // fallback kernel
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 8, 32, 1, 8, 1, 4, 1, 4, 1, 4, 4>, // fallback kernel
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 8, 32, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 4, 64, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 2, 128, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 2, 128, 1, 16, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 2, 128, 1, 32, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 32, 1, 8, 1, 8, 1, 8, 8>,
        DeviceLayernormImpl<F16, F16, F16, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 2, 1, 2, 1, 2, 2>
    // clang-format on
    >;

void add_device_layernorm_rank_2_1_f16_instances(
    std::vector<std::unique_ptr<DeviceLayernorm<F16, F16, F16, F32, F16, Pass, 2, 1>>>& instances)
{
    add_device_operation_instances(instances, device_layernorm_f16_instances<Pass, 2, 1>{});
}

void add_device_layernorm_rank_4_3_f16_instances(
    std::vector<std::unique_ptr<DeviceLayernorm<F16, F16, F16, F32, F16, Pass, 4, 3>>>& instances)
{
    add_device_operation_instances(instances, device_layernorm_f16_instances<Pass, 4, 3>{});
}

void add_device_layernorm_rank_5_3_f16_instances(
    std::vector<std::unique_ptr<DeviceLayernorm<F16, F16, F16, F32, F16, Pass, 5, 3>>>& instances)
{
    add_device_operation_instances(instances, device_layernorm_f16_instances<Pass, 5, 3>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
