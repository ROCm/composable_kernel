// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_multiple_d_softmax_gemm_permute_xdl_cshuffle_gmk_gnk_gno_gmo_instance.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using PassThrough   = ck::tensor_operation::element_wise::PassThrough;
using ScaleBiasMask = ck::tensor_operation::element_wise::ScaleBiasMask;

// f16 ScaleBiasMask masking
void add_device_batched_gemm_mutiple_d_softmax_gemm_permute_xdl_cshuffle_gmk_gnk_gno_gmo_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemmSoftmaxGemmPermute<2,
                                            1,
                                            1,
                                            1,
                                            1,
                                            F16,
                                            F16,
                                            F16,
                                            F16,
                                            ck::Tuple<F16, F16>,
                                            ck::Tuple<>,
                                            PassThrough,
                                            PassThrough,
                                            ScaleBiasMask,
                                            PassThrough,
                                            PassThrough,
                                            MaskingSpecialization::MaskOutUpperTriangle>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_gmk_gnk_gno_gmo_instances<
            2,
            1,
            1,
            1,
            1,
            F16,
            F32,
            ck::Tuple<F16, F16>,
            ScaleBiasMask,
            MaskingSpecialization::MaskOutUpperTriangle>{});
}

// f16 ScaleBiasMask disable masking
void add_device_batched_gemm_mutiple_d_softmax_gemm_permute_xdl_cshuffle_gmk_gnk_gno_gmo_instances(
    std::vector<
        std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            F16,
                                                            F16,
                                                            F16,
                                                            F16,
                                                            ck::Tuple<F16, F16>,
                                                            ck::Tuple<>,
                                                            PassThrough,
                                                            PassThrough,
                                                            ScaleBiasMask,
                                                            PassThrough,
                                                            PassThrough,
                                                            MaskingSpecialization::MaskDisabled>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_gmk_gnk_gno_gmo_instances<
            2,
            1,
            1,
            1,
            1,
            F16,
            F32,
            ck::Tuple<F16, F16>,
            ScaleBiasMask,
            MaskingSpecialization::MaskDisabled>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
