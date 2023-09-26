// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/**
 * \brief Convolution Tensor Rearrange.
 *
 * This Device operator supports conversion image ([G, N, Di, Hi, Wi, C]) to
 * the gemm problem([N * Do * Ho * Wo, Z *  Y * X * C]) (Image to Column) and
 * conversion gemm form to the image (Column to Image).
 *
 * Note that G must be equal to 1.
 *
 * \tparam NDimSpatial Number of spatial dimensions.
 * \tparam ImageLayout Input Layout.
 * \tparam InputDataType Input Data Type.
 * \tparam OutputDataType Output Data Type.
 * \tparam ConvTensorRearrangeOp Operation type: ImageToColumn, ColumnToImage.
 */
template <index_t NDimSpatial,
          typename ImageLayout,
          typename InputDataType,
          typename OutputDataType,
          typename ConvTensorRearrangeOp>
struct DeviceConvTensorRearrange : public BaseOperator
{

    /**
     * \brief Make argument pointer for image to column.
     *
     * \param p_in A pointer to the device memory of the input image.
     * \param p_out A pointer to the device memory of the output.
     * \param N Convolution batch size.
     * \param C Convolution number of channels.
     * \param input_spatial_lengths Input spatial lengths.
     * \param filter_spatial_lengths Filter spatial lengths.
     * \param output_spatial_lengths Output spatial lengths.
     * \param image_g_n_c_wis_strides Image strides in order [G, N, C, D, H, W].
     * \param gemm_m_k_strides Gemm form strides.
     * \param conv_filter_strides Convolution filter strides.
     * \param conv_filter_dilations Convolution filter dilations.
     * \param input_left_pads Convolution left pads.
     * \param input_right_pads Convolution right pads.
     * \return Pointer to the argument.
     */
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        void* p_out,
                        const ck::index_t N,
                        const ck::index_t C,
                        const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                        const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                        const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                        const std::array<index_t, NDimSpatial + 3>& image_g_n_c_wis_strides,
                        const std::array<index_t, 2>& gemm_m_k_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
