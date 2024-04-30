// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Multiply    = ck::tensor_operation::element_wise::Multiply;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;

#ifdef CK_ENABLE_INT8
// RRR
void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      AddFastGelu>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      Add>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      FastGelu>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      PassThrough>>>& instances);

// RCR
void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Col, Col>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      AddFastGelu>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Col, Col>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      Add>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Col, Col>,
                                                      ck::Tuple<>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      FastGelu>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Col, Col>,
                                                      ck::Tuple<>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      PassThrough>>>& instances);

// CRR
void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_bias_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Col>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      AddFastGelu>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_bias_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Col>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      Add>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Col>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      FastGelu>>>& instances);

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Col>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<>,
                                                      BF16,
                                                      PassThrough,
                                                      Multiply,
                                                      PassThrough>>>& instances);

// Multiply
void add_device_gemm_xdl_multi_abd_multiply_bf16_i8_bf16_mk_kn_mn_bias_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row>,
                                                      ck::Tuple<Row, Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8>,
                                                      ck::Tuple<BF16, BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      MultiplyAddFastGelu>>>& instances);

void add_device_gemm_xdl_multi_abd_multiply_bf16_i8_bf16_mk_kn_mn_bias_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8>,
                                                      ck::Tuple<BF16, BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      MultiplyAdd>>>& instances);

void add_device_gemm_xdl_multi_abd_multiply_bf16_i8_bf16_mk_kn_mn_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      MultiplyFastGelu>>>& instances);

void add_device_gemm_xdl_multi_abd_multiply_bf16_i8_bf16_mk_kn_mn_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      Multiply>>>& instances);

#endif

// GEMM + Add + Gelu
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        Multiply,
                                                        AddFastGelu>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           Multiply,
                                           AddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_gelu_v1_instances(op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Col>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_bias_gelu_v1_instances(op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Col, Col>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_gelu_v1_instances(op_ptrs);
            }
        }
#endif

        return op_ptrs;
    }
};

// GEMM + Add
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        Multiply,
                                                        Add>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           Multiply,
                                           Add>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_v1_instances(op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Col>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_bias_v1_instances(op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Col, Col>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_bias_v1_instances(op_ptrs);
            }
        }
#endif

        return op_ptrs;
    }
};

// GEMM + Gelu
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        Multiply,
                                                        FastGelu>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           Multiply,
                                           FastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_gelu_v1_instances(op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Col>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_gelu_v1_instances(op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Col, Col>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_gelu_v1_instances(op_ptrs);
            }
        }
#endif

        return op_ptrs;
    }
};

// GEMM
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        Multiply,
                                                        PassThrough>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           Multiply,
                                           PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_v1_instances(op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Col>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_v1_instances(op_ptrs);
            }

            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Col, Col>> &&
                         is_same_v<DsLayout, ck::Tuple<>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_nk_mn_v1_instances(op_ptrs);
            }
        }
#endif

        return op_ptrs;
    }
};

// Multiply
// GEMM + Add + Gelu
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        PassThrough,
                                                        MultiplyAddFastGelu>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           PassThrough,
                                           MultiplyAddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16, BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row, Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_multiply_bf16_i8_bf16_mk_kn_mn_bias_gelu_v1_instances(
                    op_ptrs);
            }
        }
#endif

        return op_ptrs;
    }
};

// GEMM + Add
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        PassThrough,
                                                        MultiplyAdd>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           PassThrough,
                                           MultiplyAdd>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16, BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row, Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_multiply_bf16_i8_bf16_mk_kn_mn_bias_v1_instances(
                    op_ptrs);
            }
        }
#endif

        return op_ptrs;
    }
};

// GEMM + Gelu
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        PassThrough,
                                                        MultiplyFastGelu>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           PassThrough,
                                           MultiplyFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_multiply_bf16_i8_bf16_mk_kn_mn_gelu_v1_instances(
                    op_ptrs);
            }
        }
#endif

        return op_ptrs;
    }
};

// GEMM
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        PassThrough,
                                                        Multiply>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           PassThrough,
                                           Multiply>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_INT8
        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_multiply_bf16_i8_bf16_mk_kn_mn_v1_instances(op_ptrs);
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
