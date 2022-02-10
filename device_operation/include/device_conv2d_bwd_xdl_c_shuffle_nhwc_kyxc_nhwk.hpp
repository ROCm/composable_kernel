#ifndef DEVICE_CONV2D_FWD_XDL_C_SHUFFLE_NHWC_KYXC_NHWK_HPP
#define DEVICE_CONV2D_FWD_XDL_C_SHUFFLE_NHWC_KYXC_NHWK_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_conv_fwd.hpp"
#include "convolution_forward_specialization.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v3r1.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <
    typename InDataType,
    typename WeiDataType,
    typename OutDataType,
    typename AccDataType,
    typename InElementwiseOperation,
    typename WeiElementwiseOperation,
    typename OutElementwiseOperation,
    ConvolutionForwardSpecialization_t ConvForwardSpecialization,
    ck::index_t BlockSize,
    ck::index_t MPerBlock,
    ck::index_t NPerBlock,
    ck::index_t K0PerBlock,
    ck::index_t K1,
    ck::index_t MPerXdl,
    ck::index_t NPerXdl,
    ck::index_t MXdlPerWave,
    ck::index_t NXdlPerWave,
    typename ABlockTransferThreadClusterLengths_K0_M_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    ck::index_t ABlockTransferSrcVectorDim,
    ck::index_t ABlockTransferSrcScalarPerVector,
    ck::index_t ABlockTransferDstScalarPerVector_K1,
    bool ABlockLdsAddExtraM,
    typename BBlockTransferThreadClusterLengths_K0_N_K1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    ck::index_t BBlockTransferSrcVectorDim,
    ck::index_t BBlockTransferSrcScalarPerVector,
    ck::index_t BBlockTransferDstScalarPerVector_K1,
    bool BBlockLdsAddExtraN,
    index_t CShuffleMXdlPerWavePerShuffle,
    index_t CShuffleNXdlPerWavePerShuffle,
    typename CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
    index_t CBlockTransferScalarPerVector_NWaveNPerXdl>
struct DeviceConv2dBwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K
    : public DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>
{
    using DeviceOp = DeviceConv2dBwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr index_t NDimSpatial = 2;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number     = Number<K1>{};
    static constexpr auto GemmK1Number = K1Number;

    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(ck::index_t N,
                                                    ck::index_t K,
                                                    ck::index_t C,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads,
                                                    index_t i_ytilda,
                                                    index_t i_xtilda)
    {
        using namespace ck;

        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t Y = filter_spatial_lengths[0];
        const index_t X = filter_spatial_lengths[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        const auto YTilda = ConvStrideH / GcdStrideDilationH;
        const auto XTilda = ConvStrideW / GcdStrideDilationW;

        const auto YDot = math::integer_divide_ceil(Y, YTilda);
        const auto XDot = math::integer_divide_ceil(X, XTilda);

        const auto HTilda = Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
        const auto WTilda = Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

        // only work on HTilda and WTilda that contribute to non-padding area of input tensor
        const auto IHTildaSliceBegin = math::integer_divide_floor(
            math::max(I0, InLeftPadH - ConvDilationH * (YTilda - I1)), ConvStrideH);
        const auto IWTildaSliceBegin = math::integer_divide_floor(
            math::max(I0, InLeftPadW - ConvDilationW * (XTilda - I1)), ConvStrideW);

        const auto IHTildaSliceEnd =
            math::min(HTilda, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
        const auto IWTildaSliceEnd =
            math::min(WTilda, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

        const auto HTildaSlice = IHTildaSliceEnd - IHTildaSliceBegin;
        const auto WTildaSlice = IWTildaSliceEnd - IWTildaSliceBegin;

        const auto K0 = K / K1;

        // GemmK is different for each GEMM
        const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilda, YTilda);
        const auto XDotSlice = math::integer_divide_ceil(X - i_xtilda, XTilda);

        // A: output tensor
        const auto out_n_ho_wo_k_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));

        const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
            out_n_ho_wo_k_grid_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(Ho, I0, I0),
                       make_pad_transform(Wo, I0, I0),
                       make_pass_through_transform(K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto out_n_ydot_htilda_xdot_wtilda_k_grid_desc = transform_tensor_descriptor(
            out_n_hop_wop_k_grid_desc,
            make_tuple(make_pass_through_transform(N),
                       make_embed_transform(make_tuple(YDot, HTilda),
                                            make_tuple(-ConvDilationH / GcdStrideDilationH, I1)),
                       make_embed_transform(make_tuple(XDot, WTilda),
                                            make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                       make_pass_through_transform(K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        const auto out_n_ydotslice_htildaslice_xdotslice_wtildaslice_k0_k1_grid_desc =
            transform_tensor_descriptor(
                out_n_ydot_htilda_xdot_wtilda_k_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_slice_transform(YDot, I0, YDotSlice),
                           make_slice_transform(HTilda, IHTildaSliceBegin, HTildaSlice),
                           make_slice_transform(XDot, I0, XDotSlice),
                           make_slice_transform(WTilda, IWTildaSliceBegin, WTildaSlice),
                           make_unmerge_transform(make_tuple(K0, K1))),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5, 6>{}));

        const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
            out_n_ydotslice_htildaslice_xdotslice_wtildaslice_k0_k1_grid_desc,
            make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                       make_merge_transform(make_tuple(N, HTildaSlice, WTildaSlice)),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}, Sequence<6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // B weight tensor
        const auto wei_k_y_x_c_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, Y, X, C));
        const auto wei_k_ydot_ytilda_xdot_xtilda_c_grid_desc = transform_tensor_descriptor(
            wei_k_y_x_c_grid_desc,
            make_tuple(make_pass_through_transform(K),
                       make_embed_transform(make_tuple(YDot, YTilda),
                                            make_tuple(ConvStrideH / GcdStrideDilationH, I1)),
                       make_embed_transform(make_tuple(XDot, XTilda),
                                            make_tuple(ConvStrideW / GcdStrideDilationW, I1)),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        const auto wei_k0_k1_ydotslice_xdotslice_c_grid_desc =
            transform_tensor_descriptor(wei_k_ydot_ytilda_xdot_xtilda_c_grid_desc,
                                        make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                   make_slice_transform(YDot, I0, YDotSlice),
                                                   make_slice_transform(XDot, I0, XDotSlice),
                                                   make_freeze_transform(i_ytilda),
                                                   make_freeze_transform(i_xtilda),
                                                   make_pass_through_transform(C)),
                                        make_tuple(Sequence<0>{},
                                                   Sequence<1>{},
                                                   Sequence<3>{},
                                                   Sequence<2>{},
                                                   Sequence<4>{},
                                                   Sequence<5>{}),
                                        make_tuple(Sequence<0, 1>{},
                                                   Sequence<2>{},
                                                   Sequence<3>{},
                                                   Sequence<>{},
                                                   Sequence<>{},
                                                   Sequence<4>{}));

        const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
            wei_k0_k1_ydotslice_xdotslice_c_grid_desc,
            make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                       make_pass_through_transform(C),
                       make_pass_through_transform(K1)),
            make_tuple(Sequence<2, 3, 0>{}, Sequence<4>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // C: input tensor
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

        const auto in_n_ytilda_htilda_xtilda_wtilda_c_grid_desc = transform_tensor_descriptor(
            in_n_hip_wip_c_grid_desc,
            make_tuple(make_pass_through_transform(N),
                       make_embed_transform(make_tuple(YTilda, HTilda),
                                            make_tuple(ConvDilationH, ConvStrideH)),
                       make_embed_transform(make_tuple(XTilda, WTilda),
                                            make_tuple(ConvDilationW, ConvStrideW)),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        const auto in_n_htildaslice_wtildaslice_c_grid_desc = transform_tensor_descriptor(
            in_n_ytilda_htilda_xtilda_wtilda_c_grid_desc,
            make_tuple(make_pass_through_transform(N),
                       make_freeze_transform(i_ytilda),
                       make_slice_transform(HTilda, IHTildaSliceBegin, HTildaSlice),
                       make_freeze_transform(i_xtilda),
                       make_slice_transform(WTilda, IWTildaSliceBegin, WTildaSlice),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<>{},
                       Sequence<1>{},
                       Sequence<>{},
                       Sequence<2>{},
                       Sequence<3>{}));

        const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
            in_n_htildaslice_wtildaslice_c_grid_desc,
            make_tuple(make_merge_transform(make_tuple(N, HTildaSlice, WTildaSlice)),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                          wei_gemmk0_gemmn_gemmk1_grid_desc,
                          in_gemmm_gemmn_grid_desc);

    } // function end

    using ABCGridDescs = decltype(MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, 0, 0));

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v3r1<
        BlockSize,
        ABDataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum_t::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        InElementwiseOperation,
        WeiElementwiseOperation,
        OutElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXdl,
        NPerXdl,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        Sequence<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder,
        2,                 // ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        Sequence<1, 0, 2>, // BBlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // BBlockTransferSrcAccessOrder,
        2,                 // BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
        CBlockTransferScalarPerVector_NWaveNPerXdl>;

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
                 ck::index_t M01,
                 ck::index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in_grid},
              p_b_grid_{p_wei_grid},
              p_c_grid_{p_out_grid},
              M01_{M01},
              N01_{N01},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op},
              Conv_N_{N},
              Conv_K_{K},
              Conv_C_{C},
              input_spatial_lengths_{input_spatial_lengths},
              filter_spatial_lengths_{filter_spatial_lengths},
              output_spatial_lengths_{output_spatial_lengths},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            const index_t ConvStrideH = conv_filter_strides[0];
            const index_t ConvStrideW = conv_filter_strides[1];

            const index_t ConvDilationH = conv_filter_dilations[0];
            const index_t ConvDilationW = conv_filter_dilations[1];

            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilda = ConvStrideH / GcdStrideDilationH;
            const auto XTilda = ConvStrideW / GcdStrideDilationW;

            for(index_t i_ytilda = 0; i_ytilda < YTilda; ++i_ytilda)
            {
                for(index_t i_xtilda = 0; i_xtilda < XTilda; ++i_xtilda)
                {
                    const auto descs = DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
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
                        i_ytilda,
                        i_xtilda);
                    a_grid_desc_k0_m_k1_.push_back(descs[I0]);
                    b_grid_desc_k0_n_k1_.push_back(descs[I1]);
                    c_grid_desc_m_n_.push_back(descs[I2]);

                    if(GridwiseGemm::CheckValidity(descs[I0], descs[I1], descs[I2], M01_, N01_))
                    {
                        c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_
                            .push_back(
                                GridwiseGemm::
                                    MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                                        descs[I2]));

                        block_2_ctile_map_.push_back(
                            GridwiseGemm::MakeBlock2CTileMap(descs[I2], M01, N01));
                    }
                }
            }
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        std::vector<AGridDesc_K0_M_K1> a_grid_desc_k0_m_k1_;
        std::vector<BGridDesc_K0_N_K1> b_grid_desc_k0_n_k1_;
        std::vector<CGridDesc_M_N> c_grid_desc_m_n_;
        std::vector<
            typename GridwiseGemm::
                CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_;
        std::vector<typename GridwiseGemm::Block2CTileMap> block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
        // for checking IsSupportedArgument()
        index_t Conv_N_;
        index_t Conv_K_;
        index_t Conv_C_;

        std::vector<ck::index_t> input_spatial_lengths_;
        std::vector<ck::index_t> filter_spatial_lengths_;
        std::vector<ck::index_t> output_spatial_lengths_;
        std::vector<ck::index_t> conv_filter_strides_;
        std::vector<ck::index_t> conv_filter_dilations_;
        std::vector<ck::index_t> input_left_pads_;
        std::vector<ck::index_t> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            nrepeat        = 1;
            float ave_time = 0;
            for(size_t i = 0; i < arg.a_grid_desc_k0_m_k1_.size(); i++)
            {
                {
                    std::cout << "arg.a_grid_desc_k0_m_k1_{"
                              << arg.a_grid_desc_k0_m_k1_[i].GetLength(I0) << ", "
                              << arg.a_grid_desc_k0_m_k1_[i].GetLength(I1) << ", "
                              << arg.a_grid_desc_k0_m_k1_[i].GetLength(I2) << "}" << std::endl;

                    std::cout << "arg.b_grid_desc_k0_n_k1_{"
                              << arg.b_grid_desc_k0_n_k1_[i].GetLength(I0) << ", "
                              << arg.b_grid_desc_k0_n_k1_[i].GetLength(I1) << ", "
                              << arg.b_grid_desc_k0_n_k1_[i].GetLength(I2) << "}" << std::endl;

                    std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_[i].GetLength(I0)
                              << ", " << arg.c_grid_desc_m_n_[i].GetLength(I1) << "}" << std::endl;

                    std::cout
                        << "arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_"
                           "nwavenperxdl_{ "
                        << arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_
                               [i]
                                   .GetLength(I0)
                        << ", "
                        << arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_
                               [i]
                                   .GetLength(I1)
                        << ", "
                        << arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_
                               [i]
                                   .GetLength(I2)
                        << ", "
                        << arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_
                               [i]
                                   .GetLength(I3)
                        << ", "
                        << arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_
                               [i]
                                   .GetLength(I4)
                        << ", "
                        << arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_
                               [i]
                                   .GetLength(I5)
                        << " } " << std::endl;
                }

            }
            return ave_time;
        }

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
             arg.Conv_C_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_K_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        for(int i = 0; i < arg.a_grid_desc_k0_m_k1_.size(); i++)
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_[i],
                                            arg.b_grid_desc_k0_n_k1_[i],
                                            arg.c_grid_desc_m_n_[i],
                                            arg.M01_,
                                            arg.N01_))
            {
                return false;
            }
        }
        return true;
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
                        1,
                        1,
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
                                          1,
                                          1,
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
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConv2dBwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
