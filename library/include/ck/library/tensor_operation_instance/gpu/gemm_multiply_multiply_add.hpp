// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_INT8))
void add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col, Row>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F32, F32, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiplyAdd>>>& instances);

void add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col, Row>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F32, F32, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiplyAdd>>>& instances);

void add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col, Row>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F32, F32, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiplyAdd>>>& instances);

void add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col, Row>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F32, F32, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiplyAdd>>>& instances);

void add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col, Row>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F32, F32, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiplyAdd>>>& instances);

void add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Tuple<Row, Col, Row>,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          Tuple<F32, F32, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiplyAdd>>>& instances);

#endif

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename DsDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemmMultipleDSplitK<
    ALayout,
    BLayout,
    Tuple<Row, Col, Row>,
    CLayout,
    ADataType,
    BDataType,
    DsDataType,
    CDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::MultiplyMultiplyAdd>>
{
    using DeviceOp =
        DeviceGemmMultipleDSplitK<ALayout,
                                  BLayout,
                                  Tuple<Row, Col, Row>,
                                  CLayout,
                                  ADataType,
                                  BDataType,
                                  DsDataType,
                                  CDataType,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::MultiplyMultiplyAdd>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#if(defined(CK_ENABLE_BF16) || defined(CK_ENABLE_INT8))
        if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_comp_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_comp_kpadding_instances(
                    op_ptrs);

                add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_mem_v1_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_mem_v1_kpadding_instances(
                    op_ptrs);

                add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_mem_v2_default_instances(
                    op_ptrs);
                add_device_gemm_multiply_multiply_add_xdl_i8_i8_bf16_mk_nk_mn_mem_v2_kpadding_instances(
                    op_ptrs);
            }
        }
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
