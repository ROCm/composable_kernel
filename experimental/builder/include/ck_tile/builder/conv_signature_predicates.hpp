// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {

/**********************************************
 * Conv Direction Predicates
 **********************************************/

// Predicate for forward convolution.
template <auto Sig>
concept ConvDirectionIsForward = (Sig.direction == ConvDirection::FORWARD);

// Predicate for backward data convolution.
template <auto Sig>
concept ConvDirectionIsBackwardData = (Sig.direction == ConvDirection::BACKWARD_DATA);

// Predicate for backward weight convolution.
template <auto Sig>
concept ConvDirectionIsBackwardWeight = (Sig.direction == ConvDirection::BACKWARD_WEIGHT);

/**********************************************
 * Conv Fwd Device Op Predicates
 **********************************************/

// Predicate for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 =
    ConvDirectionIsForward<Sig> &&
    (Sig.device_operation._fwd ==
     FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3);

// Predicate for DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK =
    ConvDirectionIsForward<Sig> &&
    (Sig.device_operation._fwd ==
     FwdGroupConvDeviceOperation::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK);

// Predicate for DeviceGroupedConvFwdMultipleD_Wmma_CShuffle operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle =
    ConvDirectionIsForward<Sig> &&
    (Sig.device_operation._fwd ==
     FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleD_Wmma_CShuffle);

// Predicate for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle =
    ConvDirectionIsForward<Sig> &&
    (Sig.device_operation._fwd ==
     FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle);

// Predicate for DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor =
    ConvDirectionIsForward<Sig> &&
    (Sig.device_operation._fwd ==
     FwdGroupConvDeviceOperation::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor);

// Generic predicate to check if signature uses any forward convolution device operation.
template <auto Sig>
concept ConvDeviceOpIsForward =
    ConvDeviceOpIs_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<Sig>;

/**********************************************
 * Conv Bwd Weight Device Op Predicates
 **********************************************/

// Predicate for DeviceGroupedConvBwdWeight operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdWeight =
    ConvDirectionIsBackwardWeight<Sig> &&
    (Sig.device_operation._bwd_weight ==
     BwdWeightGroupConvDeviceOperation::DeviceGroupedConvBwdWeight);

// Predicate for DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle =
    ConvDirectionIsBackwardWeight<Sig> &&
    (Sig.device_operation._bwd_weight ==
     BwdWeightGroupConvDeviceOperation::DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle);

// Predicate for DeviceGroupedConvBwdWeight_Xdl_CShuffle operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdWeight_Xdl_CShuffle =
    ConvDirectionIsBackwardWeight<Sig> &&
    (Sig.device_operation._bwd_weight ==
     BwdWeightGroupConvDeviceOperation::DeviceGroupedConvBwdWeight_Xdl_CShuffle);

// Predicate for DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle =
    ConvDirectionIsBackwardWeight<Sig> &&
    (Sig.device_operation._bwd_weight ==
     BwdWeightGroupConvDeviceOperation::DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle);

// Predicate for DeviceGroupedConvBwdWeight_Wmma_CShuffle operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdWeight_Wmma_CShuffle =
    ConvDirectionIsBackwardWeight<Sig> &&
    (Sig.device_operation._bwd_weight ==
     BwdWeightGroupConvDeviceOperation::DeviceGroupedConvBwdWeight_Wmma_CShuffle);

// Predicate for DeviceGroupedConvBwdWeight_Xdl_CShuffleV3 operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdWeight_Xdl_CShuffleV3 =
    ConvDirectionIsBackwardWeight<Sig> &&
    (Sig.device_operation._bwd_weight ==
     BwdWeightGroupConvDeviceOperation::DeviceGroupedConvBwdWeight_Xdl_CShuffleV3);

// Predicate for DeviceGroupedConvBwdWeightMultipleD operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdWeightMultipleD =
    ConvDirectionIsBackwardWeight<Sig> &&
    (Sig.device_operation._bwd_weight ==
     BwdWeightGroupConvDeviceOperation::DeviceGroupedConvBwdWeightMultipleD);

// Predicate for DeviceGroupedConvBwdWeight_Dl operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdWeight_Dl =
    ConvDirectionIsBackwardWeight<Sig> &&
    (Sig.device_operation._bwd_weight ==
     BwdWeightGroupConvDeviceOperation::DeviceGroupedConvBwdWeight_Dl);

// Generic predicate to check if signature uses any backward weight convolution device operation.
template <auto Sig>
concept ConvDeviceOpIsBackwardWeight =
    ConvDeviceOpIs_DeviceGroupedConvBwdWeight<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdWeight_Xdl_CShuffle<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdWeight_Wmma_CShuffle<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdWeight_Xdl_CShuffleV3<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdWeightMultipleD<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdWeight_Dl<Sig>;

/**********************************************
 * Conv Bwd Data Device Op Predicates
 **********************************************/

// Predicate for DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1 operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1 =
    ConvDirectionIsBackwardData<Sig> &&
    (Sig.device_operation._bwd_data ==
     BwdDataGroupConvDeviceOperation::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1);

// Predicate for DeviceGroupedConvBwdDataMultipleD operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdDataMultipleD =
    ConvDirectionIsBackwardData<Sig> &&
    (Sig.device_operation._bwd_data ==
     BwdDataGroupConvDeviceOperation::DeviceGroupedConvBwdDataMultipleD);

// Predicate for DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle operation.
template <auto Sig>
concept ConvDeviceOpIs_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle =
    ConvDirectionIsBackwardData<Sig> &&
    (Sig.device_operation._bwd_data ==
     BwdDataGroupConvDeviceOperation::DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle);

// Generic predicate to check if signature uses any backward data convolution device operation.
template <auto Sig>
concept ConvDeviceOpIsBackwardData =
    ConvDeviceOpIs_DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdDataMultipleD<Sig> ||
    ConvDeviceOpIs_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle<Sig>;

/**********************************************
 * Generic Device Op Predicates
 **********************************************/

// Generic predicate to check if signature uses any device operation.
template <auto Sig>
concept IsValidConvDeviceOp = ConvDeviceOpIsForward<Sig> || ConvDeviceOpIsBackwardData<Sig> ||
                              ConvDeviceOpIsBackwardWeight<Sig>;

} // namespace ck_tile::builder
