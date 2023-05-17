// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/batched_gemm_softmax_gemm_permute.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using ScaleAdd    = ck::tensor_operation::element_wise::ScaleAdd;

void add_device_batched_gemm_bias_masking_softmax_gemm_permute_xdl_cshuffle_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemmSoftmaxGemmPermute<2,
                                            1,
                                            1,
                                            1,
                                            1,
                                            BF16,
                                            BF16,
                                            BF16,
                                            BF16,
                                            ck::Tuple<BF16>,
                                            ck::Tuple<>,
                                            PassThrough,
                                            PassThrough,
                                            ScaleAdd,
                                            PassThrough,
                                            PassThrough,
                                            MaskingSpecialization::MaskOutUpperTriangle>>>&
        instances)
{
    using DeviceOp =
        DeviceBatchedGemmSoftmaxGemmPermute<2,
                                            1,
                                            1,
                                            1,
                                            1,
                                            BF16,
                                            BF16,
                                            BF16,
                                            BF16,
                                            ck::Tuple<BF16>,
                                            ck::Tuple<>,
                                            PassThrough,
                                            PassThrough,
                                            ScaleAdd,
                                            PassThrough,
                                            PassThrough,
                                            MaskingSpecialization::MaskOutUpperTriangle>;

    DeviceOperationInstanceCreator<DeviceOp>::add_device_instances(instances);
}

void add_device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instances(
    std::vector<
        std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            BF16,
                                                            BF16,
                                                            BF16,
                                                            BF16,
                                                            ck::Tuple<BF16>,
                                                            ck::Tuple<>,
                                                            PassThrough,
                                                            PassThrough,
                                                            ScaleAdd,
                                                            PassThrough,
                                                            PassThrough,
                                                            MaskingSpecialization::MaskDisabled>>>&
        instances)
{
    using DeviceOp = DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                         1,
                                                         1,
                                                         1,
                                                         1,
                                                         BF16,
                                                         BF16,
                                                         BF16,
                                                         BF16,
                                                         ck::Tuple<BF16>,
                                                         ck::Tuple<>,
                                                         PassThrough,
                                                         PassThrough,
                                                         ScaleAdd,
                                                         PassThrough,
                                                         PassThrough,
                                                         MaskingSpecialization::MaskDisabled>;
    DeviceOperationInstanceCreator<DeviceOp>::add_device_instances(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
