#ifndef DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_HPP
#define DEVICE_CONV2D_FWD_AVX2_NHWC_KYXCK8_NHWK_HPP

#include <iostream>
#include <sstream>
#include <numeric>
#include "device.hpp"
#include "device_base_cpu.hpp"
#include "device_conv_fwd_cpu.hpp"
#include "convolution_forward_specialization_cpu.hpp"
#include "common_header.hpp"
#include "../../gpu/device/tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_avx2.hpp"
#include "threadwise_gemm_avx2.hpp"
#include "threadwise_tensor_slice_transfer_avx2_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace device {

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionForwardSpecialization_t ConvForwardSpecialization,
          ConvolutionForwardGemmKSpecialization_t GemmKSpecialization,
          ConvolutionForwardBlockLoopOverSpecialization_t BlockLoopOverSpecialization,
          ck::index_t NumDimSpatial,
          ck::index_t MPerBlock, // block means data are designed to fit in cache (L1/L2/L3)
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t MPerThread,
          ck::index_t NPerThread,
          bool UseALocalBuffer,
          bool UseBLocalBuffer,
          bool UseCLocalBuffer>
struct DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K
    : public DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>
{
    using DeviceOp = DeviceConvNDFwdAvx2_Input_N_Hi_Wi_C_Weight_K_Y_X_C_K8_Output_N_Ho_Wo_K;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;

    using AElementwiseOperation = InElementwiseOperation;
    using BElementwiseOperation = WeiElementwiseOperation;
    using CElementwiseOperation = OutElementwiseOperation;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr index_t NDimSpatial = NumDimSpatial;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr bool NonTemporalStore = false;

    static constexpr auto GetBlockMNKAccessOrder()
    {
        if constexpr(BlockLoopOverSpecialization == DefaultBlockLoopOver ||
                     BlockLoopOverSpecialization == LoopOver_MNK)
            return ck::Sequence<0, 1, 2>{};
        else if constexpr(BlockLoopOverSpecialization == LoopOver_MKN)
            return ck::Sequence<0, 2, 1>{};
    }

    using BlockMNKAccessOrder = decltype(GetBlockMNKAccessOrder());

    static constexpr auto GetThreadwiseGemm_Dispatch()
    {
        if constexpr(MPerThread == 4 && NPerThread == 24)
        {
            return ck::cpu::ThreadwiseGemmAvx2_MxN_4x24_Dispatch<
                InDataType,
                WeiDataType,
                OutDataType,
                ck::tensor_layout::gemm::RowMajor,
                ck::tensor_layout::gemm::ColumnMajor,
                NonTemporalStore>{};
        }
        else if constexpr(MPerThread == 6 && NPerThread == 16)
        {
            return ck::cpu::ThreadwiseGemmAvx2_MxN_6x16_Dispatch<
                InDataType,
                WeiDataType,
                OutDataType,
                ck::tensor_layout::gemm::RowMajor,
                ck::tensor_layout::gemm::ColumnMajor,
                NonTemporalStore>{};
        }
        else
        {
            // static_assert(false, "invalid Mr/Nr");
        }
    }

    using ThreadwiseGemm_Dispatch = decltype(GetThreadwiseGemm_Dispatch());

    static auto GetWeightTensorDescriptor(ck::index_t gemm_k, ck::index_t gemm_n)
    {
        return make_naive_tensor_descriptor_packed(make_tuple(gemm_n / 8, gemm_k, 8));
    }

    static auto GetOutputTensorDescriptor(ck::index_t gemm_m, ck::index_t gemm_n)
    {
        const auto out_gemm_m_n_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(gemm_m, gemm_n));

        return out_gemm_m_n_grid_desc;
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 1, bool>::type = false>
    static auto GetInputTensorDescriptor(ck::index_t N,
                                         ck::index_t C,
                                         ck::index_t gemm_m,
                                         ck::index_t gemm_k,
                                         const std::vector<ck::index_t>& input_spatial_lengths,
                                         const std::vector<ck::index_t>& filter_spatial_lengths,
                                         const std::vector<ck::index_t>& output_spatial_lengths,
                                         const std::vector<ck::index_t>& conv_filter_strides,
                                         const std::vector<ck::index_t>& conv_filter_dilations,
                                         const std::vector<ck::index_t>& input_left_pads,
                                         const std::vector<ck::index_t>& input_right_pads)
    {
        const index_t Wi          = input_spatial_lengths[0];
        const index_t Wo          = output_spatial_lengths[0];
        const index_t ConvStrideW = conv_filter_strides[0];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            const auto in_gemm_m_k_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(gemm_m, gemm_k));

            return in_gemm_m_k_grid_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization_t::Filter1x1Pad0)
        {
            const auto in_n_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Wi, C));

            const auto in_n_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_gemm_m_k_grid_desc = transform_tensor_descriptor(
                in_n_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, Wo)), make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemm_m_k_grid_desc;
        }
        else
        {
            const index_t X             = filter_spatial_lengths[0];
            const index_t ConvDilationW = conv_filter_dilations[0];
            const index_t InLeftPadW    = input_left_pads[0];
            const index_t InRightPadW   = input_right_pads[0];

            const auto in_n_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Wi, C));

            const auto in_n_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_n_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto in_gemm_m_k_grid_desc =
                transform_tensor_descriptor(in_n_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Wo)),
                                                       make_merge_transform(make_tuple(X, C))),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemm_m_k_grid_desc;
        }
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 2, bool>::type = false>
    static auto GetInputTensorDescriptor(ck::index_t N,
                                         ck::index_t C,
                                         ck::index_t gemm_m,
                                         ck::index_t gemm_k,
                                         const std::vector<ck::index_t>& input_spatial_lengths,
                                         const std::vector<ck::index_t>& filter_spatial_lengths,
                                         const std::vector<ck::index_t>& output_spatial_lengths,
                                         const std::vector<ck::index_t>& conv_filter_strides,
                                         const std::vector<ck::index_t>& conv_filter_dilations,
                                         const std::vector<ck::index_t>& input_left_pads,
                                         const std::vector<ck::index_t>& input_right_pads)
    {
        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            const auto in_gemm_m_k_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(gemm_m, gemm_k));

            return in_gemm_m_k_grid_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization_t::Filter1x1Pad0)
        {
            const auto in_n_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            const auto in_n_ho_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_gemm_m_k_grid_desc =
                transform_tensor_descriptor(in_n_ho_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemm_m_k_grid_desc;
        }
        else
        {
            const index_t Y = filter_spatial_lengths[0];
            const index_t X = filter_spatial_lengths[1];

            const index_t ConvDilationH = conv_filter_dilations[0];
            const index_t ConvDilationW = conv_filter_dilations[1];

            const index_t InLeftPadH = input_left_pads[0];
            const index_t InLeftPadW = input_left_pads[1];

            const index_t InRightPadH = input_right_pads[0];
            const index_t InRightPadW = input_right_pads[1];

            const auto in_n_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemm_m_k_grid_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                       make_merge_transform(make_tuple(Y, X, C))),
                                            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemm_m_k_grid_desc;
        }
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 3, bool>::type = false>
    static auto GetInputTensorDescriptor(ck::index_t N,
                                         ck::index_t C,
                                         ck::index_t gemm_m,
                                         ck::index_t gemm_k,
                                         ck::index_t gemm_m_pad,
                                         const std::vector<ck::index_t>& input_spatial_lengths,
                                         const std::vector<ck::index_t>& filter_spatial_lengths,
                                         const std::vector<ck::index_t>& output_spatial_lengths,
                                         const std::vector<ck::index_t>& conv_filter_strides,
                                         const std::vector<ck::index_t>& conv_filter_dilations,
                                         const std::vector<ck::index_t>& input_left_pads,
                                         const std::vector<ck::index_t>& input_right_pads)
    {
        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            const auto in_gemm_m_k_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(gemm_m, gemm_k));

            return in_gemm_m_k_grid_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization_t::Filter1x1Pad0)
        {
            const auto in_n_di_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Di, Hi, Wi, C));

            const auto in_n_do_ho_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Do), make_tuple(ConvStrideD)),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_gemm_m_k_grid_desc = transform_tensor_descriptor(
                in_n_do_ho_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemm_m_k_grid_desc;
        }
        else
        {
            const index_t Z = filter_spatial_lengths[0];
            const index_t Y = filter_spatial_lengths[1];
            const index_t X = filter_spatial_lengths[2];

            const index_t ConvDilationD = conv_filter_dilations[0];
            const index_t ConvDilationH = conv_filter_dilations[1];
            const index_t ConvDilationW = conv_filter_dilations[2];

            const index_t InLeftPadD = input_left_pads[0];
            const index_t InLeftPadH = input_left_pads[1];
            const index_t InLeftPadW = input_left_pads[2];

            const index_t InRightPadD = input_right_pads[0];
            const index_t InRightPadH = input_right_pads[1];
            const index_t InRightPadW = input_right_pads[2];

            const auto in_n_di_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Di, Hi, Wi, C));

            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, InLeftPadD, InRightPadD),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1, 2>{},
                           Sequence<3, 4>{},
                           Sequence<5, 6>{},
                           Sequence<7>{}));

            const auto in_gemm_m_k_grid_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_merge_transform(make_tuple(Z, Y, X, C))),
                make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemm_m_k_grid_desc;
        }
    }

    static index_t GetGemmM(ck::index_t N, const std::vector<ck::index_t>& output_spatial_lengths)
    {
        return N * std::accumulate(std::begin(output_spatial_lengths),
                                   std::end(output_spatial_lengths),
                                   1,
                                   std::multiplies<ck::index_t>());
    }

    static index_t GetGemmK(ck::index_t C, const std::vector<ck::index_t>& filter_spatial_lengths)
    {
        return C * std::accumulate(std::begin(filter_spatial_lengths),
                                   std::end(filter_spatial_lengths),
                                   1,
                                   std::multiplies<ck::index_t>());
    }

    static index_t GetGemmN(ck::index_t K)
    {
        // return ck::math::integer_least_multiple(K,
        // ThreadwiseGemm_Dispatch::MatrixBMinVectorSize);
        return K;
    }

    static auto MakeABCGridDescriptor(ck::index_t N,
                                      ck::index_t K,
                                      ck::index_t C,
                                      std::vector<ck::index_t> input_spatial_lengths,
                                      std::vector<ck::index_t> filter_spatial_lengths,
                                      std::vector<ck::index_t> output_spatial_lengths,
                                      std::vector<ck::index_t> conv_filter_strides,
                                      std::vector<ck::index_t> conv_filter_dilations,
                                      std::vector<ck::index_t> input_left_pads,
                                      std::vector<ck::index_t> input_right_pads)
    {
        using namespace ck;

        const index_t GemmM = GetGemmM(N, output_spatial_lengths);
        const index_t GemmN = GetGemmN(K);
        const index_t GemmK = GetGemmK(C, filter_spatial_lengths);

        // A:
        const auto in_gemm_m_k_grid_desc =
            GetInputTensorDescriptor<NumDimSpatial>(N,
                                                    C,
                                                    GemmM,
                                                    GemmK,
                                                    input_spatial_lengths,
                                                    filter_spatial_lengths,
                                                    output_spatial_lengths,
                                                    conv_filter_strides,
                                                    conv_filter_dilations,
                                                    input_left_pads,
                                                    input_right_pads);
        // B:
        const auto wei_gemm_n0_k_n1_grid_desc = GetWeightTensorDescriptor(GemmK, GemmN);
        // C:
        const auto out_gemm_m_n_grid_desc = GetOutputTensorDescriptor(GemmM, GemmN);

        return make_tuple(
            in_gemm_m_k_grid_desc, wei_gemm_n0_k_n1_grid_desc, out_gemm_m_n_grid_desc);
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor(1, 1, 1, {1}, {1}, {1}, {1}, {1}, {1}, {1});
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor(
            1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor(
            1, 1, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NumDimSpatial>());

    using AGridDesc = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    static constexpr auto GetInputBlockDescriptor()
    {
        if constexpr(UseALocalBuffer)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(MPerBlock, KPerBlock));
        }
        else
        {
            return AGridDesc{};
        }
    }

    static constexpr auto GetWeightBlockDescriptor()
    {
        if constexpr(UseBLocalBuffer)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(
                math::integer_divide_ceil(NPerBlock, ThreadwiseGemm_Dispatch::MatrixBMinVectorSize),
                KPerBlock,
                ThreadwiseGemm_Dispatch::MatrixBMinVectorSize));
        }
        else
        {
            return BGridDesc{};
        }
    }

    static constexpr auto GetOutputBlockDescriptor()
    {
        if constexpr(UseCLocalBuffer)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(MPerBlock, NPerBlock));
        }
        else
        {
            return CGridDesc{};
        }
    }

    // static constexpr bool UseCLocalBuffer = false;

    using AThreadwiseCopy =
        ck::cpu::ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_In_NHWC<
            InDataType,
            InDataType,
            AGridDesc,
            decltype(GetInputBlockDescriptor()),
            InElementwiseOperation,
            !UseALocalBuffer,
            ConvForwardSpecialization,
            GemmKSpecialization>;

    using BThreadwiseCopy =
        ck::cpu::ThreadwiseTensorSliceTransferAvx2Specialization_ConvFwd_Wei_KYXCK8<
            WeiDataType,
            WeiDataType,
            BGridDesc,
            decltype(GetWeightBlockDescriptor()),
            WeiElementwiseOperation,
            !UseBLocalBuffer,
            ConvForwardSpecialization,
            GemmKSpecialization>;

    using CThreadwiseCopy = ck::cpu::ThreadwiseTensorSliceTransferAvx2Specialization_MatC_Store_MxN<
        OutDataType,
        OutDataType,
        CGridDesc,
        decltype(GetOutputBlockDescriptor()),
        OutElementwiseOperation,
        !UseCLocalBuffer,
        ConvForwardSpecialization,
        GemmKSpecialization>;

    using GridwiseGemm =
        ck::cpu::GridwiseGemmAvx2_MxN<InDataType,              // InDataType,
                                      WeiDataType,             // WeiDataType,
                                      OutDataType,             // OutDataType,
                                      AGridDesc,               // AGridDesc,
                                      BGridDesc,               // BGridDesc,
                                      CGridDesc,               // CGridDesc,
                                      AElementwiseOperation,   // AElementwiseOperation,
                                      BElementwiseOperation,   // BElementwiseOperation,
                                      CElementwiseOperation,   // CElementwiseOperation,
                                      MPerBlock,               // MPerBlock,
                                      NPerBlock,               // NPerBlock,
                                      KPerBlock,               // KPerBlock,
                                      ThreadwiseGemm_Dispatch, // ThreadwiseGemm_Dispatch,
                                      AThreadwiseCopy,         // AThreadwiseCopy
                                      BThreadwiseCopy,         // BThreadwiseCopy
                                      CThreadwiseCopy,         // CThreadwiseCopy
                                      BlockMNKAccessOrder,     // BlockMNKAccessOrder,
                                      ck::Sequence<0, 1>,      // ThreadMNAccessOrder
                                      UseALocalBuffer,         // UseALocalBuffer
                                      UseBLocalBuffer,         // UseBLocalBuffer
                                      UseCLocalBuffer          // UseCLocalBuffer
                                      >;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 const WeiDataType* p_wei_grid,
                 OutDataType* p_out_grid,
                 ck::index_t N,
                 ck::index_t K,
                 ck::index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in_grid},
              p_b_grid_{p_wei_grid},
              p_c_grid_{p_out_grid},
              a_grid_desc_{},
              b_grid_desc_{},
              c_grid_desc_{},
              a_element_op_{in_element_op},
              b_element_op_{wei_element_op},
              c_element_op_{out_element_op},
              Conv_N_{N},
              Conv_K_{K},
              Conv_C_{C},
              filter_spatial_lengths_{filter_spatial_lengths},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            const auto descs = DeviceOp::MakeABCGridDescriptor(N,
                                                               K,
                                                               C,
                                                               input_spatial_lengths,
                                                               filter_spatial_lengths,
                                                               output_spatial_lengths,
                                                               conv_filter_strides,
                                                               conv_filter_dilations,
                                                               input_left_pads,
                                                               input_right_pads);
            a_grid_desc_     = descs[I0];
            b_grid_desc_     = descs[I1];
            c_grid_desc_     = descs[I2];
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc a_grid_desc_;
        BGridDesc b_grid_desc_;
        CGridDesc c_grid_desc_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        // for checking IsSupportedArgument()
        index_t Conv_N_;
        index_t Conv_K_;
        index_t Conv_C_;
        std::vector<index_t> filter_spatial_lengths_;
        std::vector<index_t> conv_filter_strides_;
        std::vector<index_t> input_left_pads_;
        std::vector<index_t> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg,
                  const StreamConfig& stream_config = StreamConfig{},
                  int nrepeat                       = 1)
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_, arg.b_grid_desc_, arg.c_grid_desc_))
            {
                throw std::runtime_error("wrong! GridwiseGemmAvx2_MxN has invalid setting");
            }

            memset(arg.p_c_grid_, 0, arg.c_grid_desc_.GetElementSpaceSize());

            const auto kernel = ck::cpu::kernel_gemm_avx_mxn<GridwiseGemm,
                                                             InDataType,
                                                             WeiDataType,
                                                             OutDataType,
                                                             AGridDesc,
                                                             BGridDesc,
                                                             CGridDesc,
                                                             AElementwiseOperation,
                                                             BElementwiseOperation,
                                                             CElementwiseOperation>;

            float ave_time = 0;

            if(nrepeat != 1)
                ave_time = launch_and_time_cpu_kernel(kernel,
                                                      nrepeat,
                                                      arg.p_a_grid_,
                                                      arg.p_b_grid_,
                                                      arg.p_c_grid_,
                                                      arg.a_grid_desc_,
                                                      arg.b_grid_desc_,
                                                      arg.c_grid_desc_,
                                                      arg.a_element_op_,
                                                      arg.b_element_op_,
                                                      arg.c_element_op_);

            // TODO: this is for benchmark purpose, so last time we clear c buffer and calculate the
            // result
            memset(arg.p_c_grid_, 0, arg.c_grid_desc_.GetElementSpaceSize());

            launch_cpu_kernel(kernel,
                              arg.p_a_grid_,
                              arg.p_b_grid_,
                              arg.p_c_grid_,
                              arg.a_grid_desc_,
                              arg.b_grid_desc_,
                              arg.c_grid_desc_,
                              arg.a_element_op_,
                              arg.b_element_op_,
                              arg.c_element_op_);

            return ave_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{},
                  int nrepeat                       = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config, nrepeat);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            if(!(arg.filter_spatial_lengths_[0] == 1 && arg.filter_spatial_lengths_[1] == 1 &&
                 arg.conv_filter_strides_[0] == 1 && arg.conv_filter_strides_[1] == 1 &&
                 arg.input_left_pads_[0] == 0 && arg.input_left_pads_[1] == 0 &&
                 arg.input_right_pads_[0] == 0 && arg.input_right_pads_[1] == 0))
            {
                return false;
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization_t::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            if(!(arg.filter_spatial_lengths_[0] == 1 && arg.filter_spatial_lengths_[1] == 1 &&
                 arg.input_left_pads_[0] == 0 && arg.input_left_pads_[1] == 0 &&
                 arg.input_right_pads_[0] == 0 && arg.input_right_pads_[1] == 0))
            {
                return false;
            }
        }

        if constexpr(GemmKSpecialization ==
                         ConvolutionForwardGemmKSpecialization_t::NHWC_GemmKLoopOverC &&
                     ConvForwardSpecialization !=
                         ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            if(!(arg.Conv_C_ % KPerBlock == 0))
                return false;
        }

        if(!(arg.Conv_K_ % 8 == 0))
            return false;

        if constexpr(!UseALocalBuffer &&
                     ConvForwardSpecialization !=
                         ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            // TODO: We can support this in the future, as long as figure out how to express tensor
            // transform
            return false;
        }

        // Gridwise GEMM size
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_, arg.b_grid_desc_, arg.c_grid_desc_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in_grid,
                             const WeiDataType* p_wei_grid,
                             OutDataType* p_out_grid,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        const void* p_wei_grid,
                        void* p_out_grid,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<const WeiDataType*>(p_wei_grid),
                                          static_cast<OutDataType*>(p_out_grid),
                                          N,
                                          K,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str                 = std::stringstream();
        auto string_local_buffer = [](bool is_local_buffer) {
            if(is_local_buffer)
                return "L";
            else
                return "G";
        };
        // clang-format off
        str << "DeviceConv" << std::to_string(NumDimSpatial) 
            << "DFwdAvx2_NHWC_KYXCK8"
            <<"_FS"<< static_cast<int>(ConvForwardSpecialization)
            <<"_KS"<< static_cast<int>(GemmKSpecialization)
            <<"_BS"<< static_cast<int>(BlockLoopOverSpecialization)
            << "_BT" << MPerBlock << "x" << NPerBlock << "x" << KPerBlock
            << "_TT" << MPerThread << "x" << NPerThread 
            << "_A" << string_local_buffer(UseALocalBuffer)
            << "_B" << string_local_buffer(UseBLocalBuffer)
            << "_C" << string_local_buffer(UseCLocalBuffer)
            ;
        if constexpr (!std::is_same<OutElementwiseOperation,
                    ck::tensor_operation::cpu::element_wise::PassThrough>::value)
        {
            str << "_" << OutElementwiseOperation::Name();
        }
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck

#endif
