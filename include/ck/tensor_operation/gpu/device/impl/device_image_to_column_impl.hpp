// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_image_to_column.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_image_to_column.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InputGridDesc,
          typename InputDataType,
          typename OutputGridDesc,
          typename OutputDataType,
          typename Block2ETileMap,
          typename GridwiseImageToColumnKernel>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_image_to_column(const InputGridDesc in_grid_desc,
                               const InputDataType* __restrict__ p_in_global,
                               const OutputGridDesc out_grid_desc,
                               OutputDataType* __restrict__ p_out_global,
                               const Block2ETileMap block_2_tile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__) || defined(__gfx908__) ||             \
    defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx1030__) || defined(__gfx1100__) || \
    defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx941__) || defined(__gfx942__))
    GridwiseImageToColumnKernel::Run(
        in_grid_desc, p_in_global, out_grid_desc, p_out_global, block_2_tile_map);
#else
    ignore = in_grid_desc;
    ignore = p_in_global;
    ignore = out_grid_desc;
    ignore = p_out_global;
    ignore = block_2_tile_map;
#endif
}

// Image to column for input layout NDHWC:
//   input : input image [N, Di, Hi, Wi, C],
//   output : output image [N * Do * Ho * Wo, Z * Y * X * C]
template <index_t NDimSpatial,
          typename InputLayout,
          typename InputDataType,
          typename OutputDataType,
          index_t BlockSize,
          index_t MPerBlock,
          index_t KPerBlock,
          typename ThreadClusterLengths,
          index_t ScalarPerVector>
struct DeviceImageToColumnImpl
    : public DeviceImageToColumn<NDimSpatial, InputLayout, InputDataType, OutputDataType>
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto conv_to_gemm_transformer =
        TransformConvFwdToGemm<NDimSpatial, ConvolutionForwardSpecialization::Default>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpecialization::MKPadding, index_t, index_t, index_t>{
            MPerBlock, 0 /* NPerBlock*/, KPerBlock};

    // Use MakeADescriptor_M_K from grouped convolution forward
    static auto
    MakeInputDescriptor_M_K(const ck::index_t N,
                            const ck::index_t C,
                            const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                            const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                            const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                            const std::array<index_t, NDimSpatial + 3>& input_g_n_c_wis_strides,
                            const std::array<index_t, NDimSpatial>& conv_filter_strides,
                            const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                            const std::array<index_t, NDimSpatial>& input_left_pads,
                            const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{1};
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{1};
        std::array<index_t, NDimSpatial + 3> c_g_n_k_wos_lengths{1};

        auto copy = [](const auto& x, auto& y, index_t dst_offset) {
            std::copy(x.begin(), x.end(), y.begin() + dst_offset);
        };

        constexpr index_t spatial_offset = 3;

        copy(input_spatial_lengths, a_g_n_c_wis_lengths, spatial_offset);
        copy(filter_spatial_lengths, b_g_k_c_xs_lengths, spatial_offset);
        copy(output_spatial_lengths, c_g_n_k_wos_lengths, spatial_offset);

        // fill only significant values (C and N)
        a_g_n_c_wis_lengths[I1] = N;
        a_g_n_c_wis_lengths[I2] = C;
        b_g_k_c_xs_lengths[I2]  = C;
        c_g_n_k_wos_lengths[I1] = N;

        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<InputLayout>(
                a_g_n_c_wis_lengths,
                input_g_n_c_wis_strides,
                b_g_k_c_xs_lengths,
                {}, // not needed for A Descriptor
                c_g_n_k_wos_lengths,
                {}, // not needed for A Descriptor
                conv_filter_strides,
                conv_filter_dilations,
                input_left_pads,
                input_right_pads);

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);
        return in_gemmm_gemmk_desc;
    }

    static auto
    MakeOutDescriptor_M_K(const ck::index_t N,
                          const ck::index_t C,
                          const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                          const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                          const std::array<index_t, 2>& output_m_k_strides)
    {
        const index_t NDoHoWo =
            N * ck::accumulate_n<index_t>(
                    output_spatial_lengths.begin(), NDimSpatial, 1, std::multiplies<>());
        const index_t CZYX =
            C * ck::accumulate_n<index_t>(
                    filter_spatial_lengths.begin(), NDimSpatial, 1, std::multiplies<>());
        const auto desc_mraw_kraw = make_naive_tensor_descriptor(
            make_tuple(NDoHoWo, CZYX), make_tuple(output_m_k_strides[I0], output_m_k_strides[I1]));

        const auto desc_m_k = matrix_padder.PadADescriptor_M_K(desc_mraw_kraw);
        return desc_m_k;
    }

    using InputGridDesc =
        remove_cvref_t<decltype(MakeInputDescriptor_M_K(1, 1, {}, {}, {}, {}, {}, {}, {}, {}))>;
    using OutputGridDesc = remove_cvref_t<decltype(MakeOutDescriptor_M_K(1, 1, {}, {}, {}))>;

    using Block2ETileMap = remove_cvref_t<
        decltype(BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, KPerBlock, OutputGridDesc>(
            OutputGridDesc{}))>;

    using GridwiseImageToColumnKernel = GridwiseImageToColumn<InputGridDesc,
                                                              InputDataType,
                                                              OutputGridDesc,
                                                              OutputDataType,
                                                              BlockSize,
                                                              MPerBlock,
                                                              KPerBlock,
                                                              ThreadClusterLengths,
                                                              ScalarPerVector,
                                                              Block2ETileMap>;

    struct Argument : public BaseArgument
    {
        Argument(const void* p_in, // input image
                 void* p_out,      // output image
                 const ck::index_t N,
                 const ck::index_t C,
                 const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                 const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                 const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                 const std::array<index_t, NDimSpatial + 3>& input_g_n_c_wis_strides,
                 const std::array<index_t, 2>& output_m_k_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads)
            : C_(C),
              X_(filter_spatial_lengths[NDimSpatial - I1]),
              p_in_{static_cast<const InputDataType*>(p_in)},
              p_out_{static_cast<OutputDataType*>(p_out)},
              input_g_n_c_wis_strides_{input_g_n_c_wis_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {

            in_grid_desc_m_k_ = MakeInputDescriptor_M_K(N,
                                                        C,
                                                        input_spatial_lengths,
                                                        filter_spatial_lengths,
                                                        output_spatial_lengths,
                                                        input_g_n_c_wis_strides,

                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads);

            out_grid_desc_m_k_ = MakeOutDescriptor_M_K(
                N, C, filter_spatial_lengths, output_spatial_lengths, output_m_k_strides);
        }

        void Print() const
        {
            std::cout << in_grid_desc_m_k_ << std::endl;
            std::cout << out_grid_desc_m_k_ << std::endl;
        }

        const ck::index_t C_;
        const ck::index_t X_;

        const InputDataType* p_in_;
        OutputDataType* p_out_;

        const std::array<index_t, NDimSpatial + 3>& input_g_n_c_wis_strides_;
        const std::array<index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<index_t, NDimSpatial>& conv_filter_dilations_;
        const std::array<index_t, NDimSpatial>& input_left_pads_;
        const std::array<index_t, NDimSpatial>& input_right_pads_;

        InputGridDesc in_grid_desc_m_k_;
        OutputGridDesc out_grid_desc_m_k_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            const auto block_2_tile_map =
                BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, KPerBlock, OutputGridDesc>(
                    arg.out_grid_desc_m_k_);
            const index_t grid_size = block_2_tile_map.CalculateGridSize(arg.out_grid_desc_m_k_);
            const auto kernel       = kernel_image_to_column<InputGridDesc,
                                                       InputDataType,
                                                       OutputGridDesc,
                                                       OutputDataType,
                                                       Block2ETileMap,
                                                       GridwiseImageToColumnKernel>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(grid_size),
                                                        dim3(BlockSize),
                                                        0,
                                                        arg.in_grid_desc_m_k_,
                                                        arg.p_in_,
                                                        arg.out_grid_desc_m_k_,
                                                        arg.p_out_,
                                                        block_2_tile_map);
            return elapsed_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const Argument& arg)
    {
        using namespace tensor_layout::convolution;
        if(!(std::is_same_v<InputLayout, GNWC> || std::is_same_v<InputLayout, GNHWC> ||
             std::is_same_v<InputLayout, GNDHWC>))
        {
            return false;
        }
        if(!(NDimSpatial >= 1 && NDimSpatial <= 3))
        {
            return false;
        }

        const auto w_pad_left  = arg.input_left_pads_[NDimSpatial - I1];
        const auto w_pad_right = arg.input_right_pads_[NDimSpatial - I1];
        const auto dilation_x  = arg.conv_filter_dilations_[NDimSpatial - I1];
        const auto stride_x    = arg.conv_filter_strides_[NDimSpatial - I1];
        bool is_w_packed       = arg.input_g_n_c_wis_strides_[NDimSpatial + I2] == arg.C_;
        bool is_c_packed       = arg.input_g_n_c_wis_strides_[I2] == 1;

        // check vector acces with c not packed
        if(!is_c_packed && ScalarPerVector != 1)
            return false;
        // check vector access of filter window row (only C if C is not packed)
        if(!is_w_packed && arg.C_ % ScalarPerVector != 0)
            return false;
        // check vector access of filter window row (X * C)
        if(arg.X_ * arg.C_ % ScalarPerVector != 0)
            return false;
        // check vector access of pads (w_pad_left/w_pad_right * C)
        if(w_pad_left * arg.C_ % ScalarPerVector != 0 ||
           w_pad_right * arg.C_ % ScalarPerVector != 0)
            return false;
        // check vector access of with stride and pad
        if((w_pad_left != 0 || w_pad_right != 0) && stride_x > 1 && arg.C_ % ScalarPerVector != 0)
            return false;
        // check vector access of with dilation
        if(dilation_x > 1 && arg.C_ % ScalarPerVector != 0)
            return false;

        return GridwiseImageToColumnKernel::CheckValidity(arg.in_grid_desc_m_k_,
                                                          arg.out_grid_desc_m_k_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_in, // input image
                             void* p_out,      // output image
                             const ck::index_t N,
                             const ck::index_t C,
                             const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                             const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                             const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                             const std::array<index_t, NDimSpatial + 3>& input_g_n_c_wis_strides,
                             const std::array<index_t, 2>& output_m_k_strides,
                             const std::array<index_t, NDimSpatial>& conv_filter_strides,
                             const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                             const std::array<index_t, NDimSpatial>& input_left_pads,
                             const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        return Argument{static_cast<const InputDataType*>(p_in),
                        static_cast<OutputDataType*>(p_out),
                        N,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        input_g_n_c_wis_strides,
                        output_m_k_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in, // input image
                        void* p_out,      // output image
                        const ck::index_t N,
                        const ck::index_t C,
                        const std::array<index_t, NDimSpatial>& input_spatial_lengths,
                        const std::array<index_t, NDimSpatial>& filter_spatial_lengths,
                        const std::array<index_t, NDimSpatial>& output_spatial_lengths,
                        const std::array<index_t, NDimSpatial + 3>& input_g_n_c_wis_strides,
                        const std::array<index_t, 2>& output_m_k_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads) override
    {
        return std::make_unique<Argument>(static_cast<const InputDataType*>(p_in),
                                          static_cast<OutputDataType*>(p_out),
                                          N,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          input_g_n_c_wis_strides,
                                          output_m_k_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceImageToColumn"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << KPerBlock << ", "
            << ScalarPerVector
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
