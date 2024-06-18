// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#ifdef CK_ENABLE_INT8
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#ifdef DL_KERNELS
// Layout(A, B, C) = [Col, Row, Row]
void add_device_gemm_quantization_dl_c_shuffle_i8_i8_i8_km_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Activation_Mul_Clamp<PassThrough>>>>&
        instances);

// Layout(A, B, C) = [Col, Col, Row]
void add_device_gemm_quantization_dl_c_shuffle_i8_i8_i8_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Activation_Mul_Clamp<PassThrough>>>>&
        instances);

// Layout(A, B, C) = [Row, Row, Row]
void add_device_gemm_quantization_dl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Activation_Mul_Clamp<PassThrough>>>>&
        instances);

// Layout(A, B, C) = [Row, Col, Row]
void add_device_gemm_quantization_dl_c_shuffle_i8_i8_i8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Activation_Mul_Clamp<PassThrough>>>>&
        instances);
#endif
// Layout(A, B, C) = [Col, Row, Row]
void add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Activation_Mul_Clamp<PassThrough>>>>&
        instances);

// Layout(A, B, C) = [Col, Col, Row]
void add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Activation_Mul_Clamp<PassThrough>>>>&
        instances);

// Layout(A, B, C) = [Row, Row, Row]
void add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Activation_Mul_Clamp<PassThrough>>>>&
        instances);

// Layout(A, B, C) = [Row, Col, Row]
void add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Activation_Mul_Clamp<PassThrough>>>>&
        instances);

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename Activation>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemmMultipleD<
    ALayout,
    BLayout,
    Empty_Tuple,
    ELayout,
    ADataType,
    BDataType,
    Empty_Tuple,
    EDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    Activation_Mul_Clamp<Activation>>>
{
    using DeviceOp = DeviceGemmMultipleD<ALayout,
                                         BLayout,
                                         Empty_Tuple,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         Empty_Tuple,
                                         EDataType,
                                         ck::tensor_operation::element_wise::PassThrough,
                                         ck::tensor_operation::element_wise::PassThrough,
                                         Activation_Mul_Clamp<Activation>>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<EDataType, int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                if constexpr(is_same_v<Activation, PassThrough>)
                {
#ifdef DL_KERNELS
                    add_device_gemm_quantization_dl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(op_ptrs);
#endif
                    add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(op_ptrs);
                }
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                if constexpr(is_same_v<Activation, PassThrough>)
                {
#ifdef DL_KERNELS
                    add_device_gemm_quantization_dl_c_shuffle_i8_i8_i8_mk_nk_mn_instances(op_ptrs);
#endif
                    add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_mk_nk_mn_instances(op_ptrs);
                }
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                if constexpr(is_same_v<Activation, PassThrough>)
                {
#ifdef DL_KERNELS
                    add_device_gemm_quantization_dl_c_shuffle_i8_i8_i8_km_kn_mn_instances(op_ptrs);
#endif
                    add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instances(op_ptrs);
                }
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                if constexpr(is_same_v<Activation, PassThrough>)
                {
#ifdef DL_KERNELS
                    add_device_gemm_quantization_dl_c_shuffle_i8_i8_i8_km_nk_mn_instances(op_ptrs);
#endif
                    add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_km_nk_mn_instances(op_ptrs);
                }
            }

            return op_ptrs;
        }
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif