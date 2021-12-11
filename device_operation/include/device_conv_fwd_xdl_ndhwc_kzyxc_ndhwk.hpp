#ifndef DEVICE_CONV3D_FWD_XDL_HPP
#define DEVICE_CONV3D_FWD_XDL_HPP

#include <iostream>
#include "device.hpp"

#include "device_conv.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "transform_forward_convolution3d_into_gemm_v4r4r4_ndhwc_kzyxc_ndhwk.hpp"
#include "gridwise_batched_gemm_xdlops_v2r3.hpp"
#include "device_conv_fwd_xdl.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// specialization for #D conv: in[n, di, hi, wi, c] * wei[k, z, y, x, c] = out[n, do, ho, wo, k]
template <typename InDataType,
          typename WeiDataType, // WeiDataType must be the same as InDataType
          typename OutDataType,
          typename AccDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadSliceLengths_K0_M_K1,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          typename BBlockTransferThreadSliceLengths_K0_N_K1,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          bool ABlockLdsAddExtraM,
          bool BBlockLdsAddExtraN>
struct DeviceConvFwdXdl<3,
                        InDataType,
                        WeiDataType, // WeiDataType must be the same as InDataType
                        OutDataType,
                        AccDataType,
                        InLayout,
                        WeiLayout,
                        OutLayout,
                        InElementwiseOperation,
                        WeiElementwiseOperation,
                        OutElementwiseOperation,
                        BlockSize,
                        MPerBlock,
                        NPerBlock,
                        K0PerBlock,
                        K1,
                        MPerXDL,
                        NPerXDL,
                        MXdlPerWave,
                        NXdlPerWave,
                        ABlockTransferThreadSliceLengths_K0_M_K1,
                        ABlockTransferThreadClusterLengths_K0_M_K1,
                        ABlockTransferThreadClusterArrangeOrder,
                        ABlockTransferSrcAccessOrder,
                        ABlockTransferSrcVectorDim,
                        ABlockTransferSrcScalarPerVector,
                        ABlockTransferDstScalarPerVector_K1,
                        BBlockTransferThreadSliceLengths_K0_N_K1,
                        BBlockTransferThreadClusterLengths_K0_N_K1,
                        BBlockTransferThreadClusterArrangeOrder,
                        BBlockTransferSrcAccessOrder,
                        BBlockTransferSrcVectorDim,
                        BBlockTransferSrcScalarPerVector,
                        BBlockTransferDstScalarPerVector_K1,
                        CThreadTransferSrcDstVectorDim,
                        CThreadTransferDstScalarPerVector,
                        ABlockLdsAddExtraM,
                        BBlockLdsAddExtraN>
    : public DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>

{
    using DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>::ComputeOutputSpatialLengths;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static auto MakeGridDescriptors(const index_t N,
                                    const index_t K,
                                    const index_t C,
                                    std::vector<ck::index_t> input_spatial_lengths,
                                    std::vector<ck::index_t> filter_spatial_lengths,
                                    std::vector<ck::index_t> conv_strides,
                                    std::vector<ck::index_t> conv_dilations,
                                    std::vector<ck::index_t> in_left_pads,
                                    std::vector<ck::index_t> in_right_pads)
    {
        assert(input_spatial_lengths.size() > 2);
        assert(filter_spatial_lengths.size() > 2);
        assert(conv_strides.size() > 2);
        assert(conv_dilations.size() > 2);
        assert(in_left_pads.size() > 2);
        assert(in_right_pads.size() > 2);

        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];
        const index_t Z  = filter_spatial_lengths[0];
        const index_t Y  = filter_spatial_lengths[1];
        const index_t X  = filter_spatial_lengths[2];

        const index_t ZEff = (filter_spatial_lengths[0] - 1) * conv_dilations[0] + 1;
        const index_t YEff = (filter_spatial_lengths[1] - 1) * conv_dilations[1] + 1;
        const index_t XEff = (filter_spatial_lengths[2] - 1) * conv_dilations[2] + 1;

        const index_t Do = (Di + in_left_pads[0] + in_right_pads[0] - ZEff) / conv_strides[0] + 1;
        const index_t Ho = (Hi + in_left_pads[1] + in_right_pads[1] - YEff) / conv_strides[1] + 1;
        const index_t Wo = (Wi + in_left_pads[2] + in_right_pads[2] - XEff) / conv_strides[2] + 1;

        const auto in_desc_n_di_hi_wi_c =
            make_naive_tensor_descriptor_packed<true>(make_tuple(N, Di, Hi, Wi, C));
        const auto wei_desc_k_z_y_x_c =
            make_naive_tensor_descriptor_packed<true>(make_tuple(K, Z, Y, X, C));
        const auto out_desc_n_do_ho_wo_k =
            make_naive_tensor_descriptor_packed<true>(make_tuple(N, Do, Ho, Wo, K));

        static_assert(is_same_v<tensor_layout::convolution::NDHWC, InLayout> &&
                      is_same_v<tensor_layout::convolution::KZYXC, WeiLayout> &&
                      is_same_v<tensor_layout::convolution::NDHWK, OutLayout>);

        if constexpr(is_same_v<tensor_layout::convolution::NDHWC, InLayout> &&
                     is_same_v<tensor_layout::convolution::KZYXC, WeiLayout> &&
                     is_same_v<tensor_layout::convolution::NDHWK, OutLayout>)
        {
            const auto descs =
                transform_forward_convolution3d_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad_split_batch(
                    in_desc_n_di_hi_wi_c,
                    wei_desc_k_z_y_x_c,
                    out_desc_n_do_ho_wo_k,
                    make_tuple(conv_strides[0], conv_strides[1], conv_strides[2]),
                    make_tuple(conv_dilations[0], conv_dilations[1], conv_dilations[2]),
                    make_tuple(in_left_pads[0], in_left_pads[1], in_left_pads[2]),
                    make_tuple(in_right_pads[0], in_right_pads[1], in_right_pads[2]),
                    Number<K1>{});

            return descs;
        }
        else
        {
            // only NDHWC_KZYXC_NDHWK layout is suppprt so far
            return;
        }
    }

    using AGridDesc_B_K0_M_K1 = remove_cvref_t<decltype(MakeGridDescriptors(
        1, 1, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1})[I0])>;
    using BGridDesc_K0_N_K1   = remove_cvref_t<decltype(MakeGridDescriptors(
        1, 1, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1})[I1])>;
    using CGridDesc_B_M_N     = remove_cvref_t<decltype(MakeGridDescriptors(
        1, 1, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1})[I2])>;

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    static constexpr index_t NumTransformsOfData = 21;
    static constexpr auto in_gemmk0_gemmm_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{}),
                   make_tuple(uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfData, 0>::type{}));

    static constexpr auto wei_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: GemmN
                              Sequence<0, 0, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: GemmN
                              Sequence<0, 0, 0, 0, 0>{})); // 2-: GemmK1

    static constexpr index_t NumTransformsOfOutput = 23;
    static constexpr auto out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
        make_tuple(make_tuple(uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{}),
                   make_tuple(uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{},
                              uniform_sequence_gen<NumTransformsOfOutput, 0>::type{}));

    static constexpr auto in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        uniform_sequence_gen<NumTransformsOfData, 0>::type{};

    static constexpr auto wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    using GridwiseBatchedGemm = GridwiseBatchedGemm_bk0mk1_k0nk1_bmn_xdlops_v2r3<
        BlockSize,
        InDataType,
        AccDataType,
        OutDataType,
        InMemoryDataOperationEnum_t::Set,
        AGridDesc_B_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_B_M_N,
        InElementwiseOperation,
        WeiElementwiseOperation,
        OutElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadSliceLengths_K0_M_K1,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        BBlockTransferThreadSliceLengths_K0_N_K1,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        decltype(in_gemmk0_gemmm_gemmk1_grid_step_hacks),
        decltype(wei_gemmk0_gemmn_gemmk1_grid_step_hacks),
        decltype(out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
        decltype(in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
        decltype(wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
        false, // CAccessOrderMRepeatNRepeat,
        ABlockLdsAddExtraM,
        BBlockLdsAddExtraN>;

    using CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2 = decltype(
        GridwiseBatchedGemm::MakeCGridDescriptor_B_M0_N0_M1_N1_M2_M3_M4_N2(CGridDesc_B_M_N{}));
    using Block2CTileMap =
        decltype(GridwiseBatchedGemm::MakeBlock2CTileMap(CGridDesc_B_M_N{}, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in,
                 const WeiDataType* p_wei,
                 OutDataType* p_out,
                 const index_t N,
                 const index_t K,
                 const index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> conv_strides,
                 std::vector<ck::index_t> conv_dilations,
                 std::vector<ck::index_t> in_left_pads,
                 std::vector<ck::index_t> in_right_pads,
                 index_t M01,
                 index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in},
              p_b_grid_{p_wei},
              p_c_grid_{p_out},
              M01_{M01},
              N01_{N01},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
            const auto descs = MakeGridDescriptors(N,
                                                   K,
                                                   C,
                                                   input_spatial_lengths,
                                                   filter_spatial_lengths,
                                                   conv_strides,
                                                   conv_dilations,
                                                   in_left_pads,
                                                   in_right_pads);

            a_grid_desc_b_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_   = descs[I1];
            c_grid_desc_b_m_n_     = descs[I2];

            a_batch_stride_ = a_grid_desc_b_k0_m_k1_.CalculateOffset(make_multi_index(1, 0, 0, 0)) -
                              a_grid_desc_b_k0_m_k1_.CalculateOffset(make_multi_index(0, 0, 0, 0));
            c_batch_stride_ = c_grid_desc_b_m_n_.CalculateOffset(make_multi_index(1, 0, 0)) -
                              c_grid_desc_b_m_n_.CalculateOffset(make_multi_index(0, 0, 0));

            if(GridwiseBatchedGemm::CheckValidity(
                   a_grid_desc_b_k0_m_k1_, b_grid_desc_k0_n_k1_, c_grid_desc_b_m_n_, M01_, N01_))
            {
                c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseBatchedGemm::MakeCGridDescriptor_B_M0_N0_M1_N1_M2_M3_M4_N2(
                        c_grid_desc_b_m_n_);

                block_2_ctile_map_ =
                    GridwiseBatchedGemm::MakeBlock2CTileMap(c_grid_desc_b_m_n_, M01, N01);
            }
        }

        //  private:
        const InDataType* p_a_grid_;
        const WeiDataType* p_b_grid_;
        OutDataType* p_c_grid_;
        int a_batch_stride_;
        int c_batch_stride_;
        AGridDesc_B_K0_M_K1 a_grid_desc_b_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_B_M_N c_grid_desc_b_m_n_;
        CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceConvFwdXdl::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            {
                std::cout << "a_grid_desc_b_k0_m_k1{" << arg.a_grid_desc_b_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_b_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_b_k0_m_k1_.GetLength(I2) << ", "
                          << arg.a_grid_desc_b_k0_m_k1_.GetLength(I3) << "}" << std::endl;

                std::cout << "b_grid_desc_k0_n_k1{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "c_grid_desc_b_m_n{ " << arg.c_grid_desc_b_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_b_m_n_.GetLength(I1) << ", "
                          << arg.c_grid_desc_b_m_n_.GetLength(I2) << "}" << std::endl;
            }

            if(!GridwiseBatchedGemm::CheckValidity(arg.a_grid_desc_b_k0_m_k1_,
                                                   arg.b_grid_desc_k0_n_k1_,
                                                   arg.c_grid_desc_b_m_n_,
                                                   arg.M01_,
                                                   arg.N01_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseBatchedGemm_bk0mk1_k0nk1_bmn_xdlops_v2r3_xdlops_v2r3 has "
                    "invalid setting");
            }

            const index_t grid_size =
                GridwiseBatchedGemm::CalculateGridSize(arg.c_grid_desc_b_m_n_);

            const auto K0 = arg.a_grid_desc_b_k0_m_k1_.GetLength(I1);

            const bool has_main_k0_block_loop =
                GridwiseBatchedGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;
            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                    GridwiseBatchedGemm,
                    InDataType,
                    OutDataType,
                    remove_reference_t<AGridDesc_B_K0_M_K1>,
                    remove_reference_t<BGridDesc_K0_N_K1>,
                    remove_reference_t<CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<Block2CTileMap>,
                    true>;
                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_batch_stride_,
                                                  arg.c_batch_stride_,
                                                  arg.a_grid_desc_b_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                    GridwiseBatchedGemm,
                    InDataType,
                    OutDataType,
                    remove_reference_t<AGridDesc_B_K0_M_K1>,
                    remove_reference_t<BGridDesc_K0_N_K1>,
                    remove_reference_t<CGridDesc_B_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<Block2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_batch_stride_,
                                                  arg.c_batch_stride_,
                                                  arg.a_grid_desc_b_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_b_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
                                                  arg.block_2_ctile_map_);
            }

            return ave_time;
        }

        // polymorphic
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
        return GridwiseBatchedGemm::CheckValidity(arg.a_grid_desc_b_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_b_m_n_,
                                                  arg.M01_,
                                                  arg.N01_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in,
                             const WeiDataType* p_wei,
                             OutDataType* p_out,
                             const index_t N,
                             const index_t K,
                             const index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> conv_strides,
                             std::vector<ck::index_t> conv_dilations,
                             std::vector<ck::index_t> in_left_pads,
                             std::vector<ck::index_t> in_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in,
                        p_wei,
                        p_out,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads,
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        const index_t N,
                        const index_t K,
                        const index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> conv_strides,
                        std::vector<ck::index_t> conv_dilations,
                        std::vector<ck::index_t> in_left_pads,
                        std::vector<ck::index_t> in_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override

    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in),
                                          static_cast<const WeiDataType*>(p_wei),
                                          static_cast<OutDataType*>(p_out),
                                          N,
                                          K,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          conv_strides,
                                          conv_dilations,
                                          in_left_pads,
                                          in_right_pads,
                                          1,
                                          1,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
